"""Unit tests for vectormind.training.logger."""

from __future__ import annotations

from vectormind.training.logger import TrainingLogger


class TestTrainingLogger:
    """Tests for TrainingLogger."""

    def test_creates_log_dir(self, tmp_path: object) -> None:
        """Creates the log directory if it doesn't exist."""
        log_dir = tmp_path / "logs" / "test"  # type: ignore[union-attr]
        logger = TrainingLogger(log_dir)
        assert log_dir.exists()
        logger.close()

    def test_log_metrics(self, tmp_path: object) -> None:
        """log_metrics does not crash."""
        log_dir = tmp_path / "logs"  # type: ignore[union-attr]
        logger = TrainingLogger(log_dir)
        logger.log_metrics(step=0, metrics={"loss": 1.0, "temp": 0.5})
        logger.close()

    def test_log_epoch(self, tmp_path: object) -> None:
        """log_epoch does not crash."""
        log_dir = tmp_path / "logs"  # type: ignore[union-attr]
        logger = TrainingLogger(log_dir)
        logger.log_epoch(epoch=0, metrics={"loss": 1.0})
        logger.close()

    def test_flush(self, tmp_path: object) -> None:
        """flush does not crash."""
        log_dir = tmp_path / "logs"  # type: ignore[union-attr]
        logger = TrainingLogger(log_dir)
        logger.log_metrics(step=0, metrics={"loss": 1.0})
        logger.flush()
        logger.close()

    def test_close(self, tmp_path: object) -> None:
        """close does not crash."""
        log_dir = tmp_path / "logs"  # type: ignore[union-attr]
        logger = TrainingLogger(log_dir)
        logger.close()

    def test_creates_event_files(self, tmp_path: object) -> None:
        """Writing metrics creates TensorBoard event files."""
        log_dir = tmp_path / "logs"  # type: ignore[union-attr]
        logger = TrainingLogger(log_dir)
        logger.log_metrics(step=0, metrics={"loss": 1.0})
        logger.flush()
        logger.close()

        # Check that event files were created
        event_files = list(log_dir.glob("events.out.tfevents.*"))
        assert len(event_files) > 0
