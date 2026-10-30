from collections.abc import Mapping
from uuid import UUID

from langgraph.checkpoint.base import BaseCheckpointSaver

from self_healing_rag.application.ports.run_inspection import (
    RunInspection,
    RunInspectionReader,
)


class CheckpointRunInspectionReader(RunInspectionReader):
    def __init__(
        self,
        checkpointer: BaseCheckpointSaver,
    ) -> None:
        self._checkpointer = checkpointer

    def get_run_inspection(
        self,
        run_id: UUID,
    ) -> RunInspection | None:
        config = {
            "configurable": {
                "thread_id": str(run_id),
            }
        }

        saved = self._checkpointer.get_tuple(config)

        if saved is None:
            return None

        channel_values = saved.checkpoint.get("channel_values", {})

        if not isinstance(channel_values, Mapping):
            raise TypeError("Invalid checkpoint channel values.")

        evaluation = channel_values.get("evaluation")

        if not isinstance(evaluation, Mapping):
            raise TypeError("Checkpoint has no valid retrieval evaluation.")

        retrieval_success = evaluation.get("success")

        if not isinstance(retrieval_success, bool):
            raise TypeError("Invalid retrieval evaluation success value.")

        retrieved_chunks = channel_values.get("retrieved_chunks", [])

        if not isinstance(retrieved_chunks, list):
            raise TypeError("Invalid retrieved chunks in checkpoint.")

        top_score = evaluation.get("top_score")

        if top_score is not None:
            if isinstance(top_score, bool) or not isinstance(
                top_score,
                (int, float),
            ):
                raise TypeError("Invalid top retrieval score.")

            top_score = float(top_score)

        minimum_score = channel_values.get("min_relevance_score")

        if isinstance(minimum_score, bool) or not isinstance(
            minimum_score,
            (int, float),
        ):
            raise TypeError("Invalid minimum relevance score.")

        retry_count = channel_values.get("retry_count", 0)

        if isinstance(retry_count, bool) or not isinstance(
            retry_count,
            int,
        ):
            raise TypeError("Invalid retry count.")

        failure_type = evaluation.get("failure_type")

        if failure_type is not None and not isinstance(
            failure_type,
            str,
        ):
            raise TypeError("Invalid failure type.")

        diagnosis = channel_values.get("diagnosis")

        if diagnosis is not None and not isinstance(
            diagnosis,
            Mapping,
        ):
            raise TypeError("Invalid diagnosis in checkpoint.")

        recovery_decision = channel_values.get("recovery_decision")

        if recovery_decision is not None and not isinstance(
            recovery_decision,
            Mapping,
        ):
            raise TypeError("Invalid recovery decision in checkpoint.")

        recommended_action = (
            diagnosis.get("recommended_action") if diagnosis is not None else None
        )

        recovery_action = (
            recovery_decision.get("action") if recovery_decision is not None else None
        )

        if recommended_action is not None and not isinstance(
            recommended_action,
            str,
        ):
            raise TypeError("Invalid recommended recovery action.")

        if recovery_action is not None and not isinstance(
            recovery_action,
            str,
        ):
            raise TypeError("Invalid recovery action.")

        answer = channel_values.get("answer")

        if answer is not None and not isinstance(answer, str):
            raise TypeError("Invalid answer in checkpoint.")

        return RunInspection(
            run_id=run_id,
            retrieval_success=retrieval_success,
            failure_type=failure_type,
            retrieved_chunk_count=len(retrieved_chunks),
            top_score=top_score,
            minimum_relevance_score=float(minimum_score),
            recommended_action=recommended_action,
            recovery_action=recovery_action,
            retry_count=retry_count,
            answer_generated=answer is not None,
        )
