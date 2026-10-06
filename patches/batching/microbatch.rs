//! Bounded, opt-in request coalescing. One worker owns inference at a time.
use super::{AnyPrefillForward, PrefillRouter};
use std::sync::Arc;
use std::time::{Duration, Instant};
use switchyard_protocol::ModelId;
use tokio::sync::{Mutex, OnceCell, mpsc, oneshot};

type Prediction = std::result::Result<Vec<(ModelId, f32)>, String>;
struct Job {
    prompt: String,
    reply: oneshot::Sender<Prediction>,
    enqueued: Instant,
}

pub(super) struct Microbatch {
    sender: OnceCell<mpsc::Sender<Job>>,
    router: Arc<Mutex<PrefillRouter<AnyPrefillForward>>>,
    size: usize,
    wait: Duration,
}

impl Microbatch {
    pub(super) fn new(
        router: Arc<Mutex<PrefillRouter<AnyPrefillForward>>>,
        size: usize,
        wait: Duration,
    ) -> Self {
        assert!(size > 0 && size <= 64);
        Self {
            sender: OnceCell::new(),
            router,
            size,
            wait,
        }
    }

    pub(super) async fn predict(&self, prompt: String) -> Prediction {
        let sender = self
            .sender
            .get_or_init(|| async {
                let (tx, rx) = mpsc::channel(128);
                tokio::spawn(worker(Arc::clone(&self.router), rx, self.size, self.wait));
                tx
            })
            .await;
        let (reply, result) = oneshot::channel();
        sender
            .try_send(Job {
                prompt,
                reply,
                enqueued: Instant::now(),
            })
            .map_err(|error| format!("prefill queue unavailable: {error}"))?;
        result
            .await
            .map_err(|error| format!("prefill worker stopped: {error}"))?
    }
}

async fn worker(
    router: Arc<Mutex<PrefillRouter<AnyPrefillForward>>>,
    mut rx: mpsc::Receiver<Job>,
    size: usize,
    wait: Duration,
) {
    while let Some(first) = rx.recv().await {
        if first.reply.is_closed() {
            continue;
        }
        let collect_started = Instant::now();
        let mut batch = vec![first];
        let deadline = tokio::time::Instant::now() + wait;
        while batch.len() < size {
            match tokio::time::timeout_at(deadline, rx.recv()).await {
                Ok(Some(job)) => {
                    if !job.reply.is_closed() {
                        batch.push(job);
                    }
                }
                _ => break,
            }
        }
        batch.retain(|job| !job.reply.is_closed());
        if batch.is_empty() {
            continue;
        }
        let inference_ready = Instant::now();
        let queue_wait_ms: Vec<f64> = batch
            .iter()
            .map(|job| inference_ready.duration_since(job.enqueued).as_secs_f64() * 1000.0)
            .collect();
        let queue_wait_mean_ms = queue_wait_ms.iter().sum::<f64>() / queue_wait_ms.len() as f64;
        let queue_wait_max_ms = queue_wait_ms.iter().copied().fold(0.0, f64::max);
        let batch_collect_ms = inference_ready.duration_since(collect_started).as_secs_f64() * 1000.0;
        let prompts: Vec<String> = batch.iter().map(|job| job.prompt.clone()).collect();
        let lock_started = Instant::now();
        let mut locked = Arc::clone(&router).lock_owned().await;
        let lock_wait_ms = lock_started.elapsed().as_secs_f64() * 1000.0;
        let predict_started = Instant::now();
        let result = tokio::task::spawn_blocking(move || {
            locked
                .predict_batch(&prompts)
                .map_err(|error| error.to_string())
        })
        .await
        .unwrap_or_else(|error| Err(format!("prefill batch task failed: {error}")));
        let predict_ms = predict_started.elapsed().as_secs_f64() * 1000.0;
        tracing::info!(
            batch_size = batch.len(),
            batch_collect_ms,
            queue_wait_mean_ms,
            queue_wait_max_ms,
            lock_wait_ms,
            predict_ms,
            "prefill request batch completed"
        );
        match result {
            Ok(values) if values.len() == batch.len() => {
                for (job, value) in batch.into_iter().zip(values) {
                    let _ = job.reply.send(Ok(value));
                }
            }
            other => {
                let message = match other {
                    Err(error) => error,
                    Ok(_) => "prefill batch returned wrong result count".to_string(),
                };
                for job in batch {
                    let _ = job.reply.send(Err(message.clone()));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PrefillForward, PrefillRouterError};
    use std::sync::Mutex as StdMutex;

    struct Mock(Arc<StdMutex<Vec<usize>>>);
    impl PrefillForward for Mock {
        fn output_count(&self) -> usize {
            1
        }
        fn forward(&mut self, prompts: &[String]) -> crate::Result<Vec<Vec<f32>>> {
            self.0.lock().unwrap().push(prompts.len());
            if prompts.iter().any(|p| p == "fail") {
                return Err(PrefillRouterError::InvalidRequest("test failure".into()));
            }
            Ok(prompts
                .iter()
                .map(|p| vec![p.parse::<f32>().unwrap() / 100.0])
                .collect())
        }
        fn unload(&mut self) -> crate::Result<()> {
            Ok(())
        }
    }
    fn make(size: usize) -> (Arc<Microbatch>, Arc<StdMutex<Vec<usize>>>) {
        let calls = Arc::new(StdMutex::new(Vec::new()));
        let router = PrefillRouter::new(
            vec![ModelId::from("model")],
            AnyPrefillForward(Box::new(Mock(calls.clone()))),
        )
        .unwrap();
        (
            Arc::new(Microbatch::new(
                Arc::new(Mutex::new(router)),
                size,
                Duration::from_millis(5),
            )),
            calls,
        )
    }
    #[tokio::test]
    async fn batches_and_preserves_request_mapping() {
        let (batcher, calls) = make(4);
        let mut tasks = tokio::task::JoinSet::new();
        for i in 0..12 {
            let batcher = batcher.clone();
            tasks.spawn(async move { (i, batcher.predict(i.to_string()).await.unwrap()) });
        }
        while let Some(result) = tasks.join_next().await {
            let (i, scores) = result.unwrap();
            assert_eq!(scores[0].1, i as f32 / 100.0);
        }
        let calls = calls.lock().unwrap();
        assert!(calls.iter().all(|n| *n <= 4));
        assert!(calls.iter().any(|n| *n > 1));
        assert_eq!(calls.iter().sum::<usize>(), 12);
    }
    #[tokio::test]
    async fn error_does_not_kill_worker() {
        let (batcher, _) = make(4);
        assert!(batcher.predict("fail".into()).await.is_err());
        assert_eq!(batcher.predict("5".into()).await.unwrap()[0].1, 0.05);
    }
    #[tokio::test]
    async fn single_request_flushes_without_waiting_for_full_batch() {
        let (batcher, _) = make(16);
        let result =
            tokio::time::timeout(Duration::from_secs(1), batcher.predict("1".into())).await;
        assert_eq!(result.unwrap().unwrap()[0].1, 0.01);
    }
}
