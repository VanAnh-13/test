"""Append-only artifact ledger dùng chung cho journey graph và persistence adapter."""

from __future__ import annotations

from collections.abc import Iterator

from hagent.agent.journey.artifacts import Artifact


class ArtifactLedger:
    """Giữ artifact theo identity; không cung cấp API update tại chỗ."""

    def __init__(self) -> None:
        self._artifacts: dict[str, Artifact] = {}
        self._children_by_supersedes: dict[str, list[str]] = {}

    def __len__(self) -> int:
        return len(self._artifacts)

    def __iter__(self) -> Iterator[Artifact]:
        return iter(self._artifacts.values())

    def append(self, artifact: Artifact) -> None:
        if artifact.artifact_id in self._artifacts:
            raise ValueError(f"Artifact already exists: {artifact.artifact_id}")
        missing_lineage = [
            parent_id for parent_id in artifact.lineage if parent_id not in self._artifacts
        ]
        if missing_lineage:
            raise ValueError("Artifact lineage references an unknown parent")
        for parent_id in artifact.lineage:
            parent = self._artifacts[parent_id]
            if parent.owner_id != artifact.owner_id or parent.run_id != artifact.run_id:
                raise ValueError("Artifact lineage must stay inside one owner and run")

        if artifact.version == 1 and artifact.supersedes is not None:
            raise ValueError("Artifact version 1 must not supersede another artifact")
        if artifact.version > 1 and artifact.supersedes is None:
            raise ValueError("Artifact revision must declare supersedes")
        if artifact.supersedes is not None:
            previous = self._artifacts.get(artifact.supersedes)
            if previous is None:
                raise ValueError("Artifact supersedes references an unknown artifact")
            if self._children_by_supersedes.get(artifact.supersedes):
                raise ValueError("Artifact already has a revision")
            if type(previous) is not type(artifact):
                raise ValueError("Artifact revision must supersede the same artifact type")
            if previous.owner_id != artifact.owner_id or previous.run_id != artifact.run_id:
                raise ValueError("Artifact revision must stay inside one owner and run")
            if artifact.version != previous.version + 1:
                raise ValueError("Artifact revision version must increment by one")

        self._artifacts[artifact.artifact_id] = artifact
        if artifact.supersedes is not None:
            self._children_by_supersedes.setdefault(artifact.supersedes, []).append(
                artifact.artifact_id
            )

    def get(self, artifact_id: str) -> Artifact:
        try:
            return self._artifacts[artifact_id]
        except KeyError as exc:
            raise KeyError(f"Unknown artifact: {artifact_id}") from exc

    def latest_revision(self, artifact_id: str) -> Artifact:
        current = self.get(artifact_id)
        while self._children_by_supersedes.get(current.artifact_id):
            child_ids = self._children_by_supersedes[current.artifact_id]
            current = max(
                (self._artifacts[child_id] for child_id in child_ids),
                key=lambda artifact: artifact.version,
            )
        return current
