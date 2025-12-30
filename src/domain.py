from dataclasses import dataclass, field


@dataclass(frozen=True)
class ImageId:
    """Lightweight identifier for an image (currently just the filename).
    In future this could be a hash or relative path if directories are nested.
    """

    filename: str


@dataclass
class ImageItem:
    id: ImageId
    path: str
    exif: dict[str, object] = field(default_factory=dict)
    rating: int = 0
    tags: list[str] = field(default_factory=list)
    label: str | None = None  # keep/trash
    objects: list[str] = field(default_factory=list)  # detected objects (person, cat, car, etc.)
    # Grouping fields
    session_id: int | None = None
    burst_id: int | None = None
    group_id: int | None = None  # Final grouping (burst or near-duplicate cluster)
    group_size: int = 1  # Number of photos in this group
    is_group_best: bool = False  # Is this the best pick in the group?
    pick_score: float = 0.0  # Heuristic score for best-pick recommendation

    @classmethod
    def from_raw(cls, filename: str, path: str, details: dict):
        return cls(
            id=ImageId(filename),
            path=path,
            exif=details.get("exif", {}),
            rating=details.get("rating", 0),
            tags=details.get("tags", []),
            label=details.get("label"),
            objects=details.get("objects", []),
            session_id=details.get("session_id"),
            burst_id=details.get("burst_id"),
            group_id=details.get("group_id"),
            group_size=details.get("group_size", 1),
            is_group_best=details.get("is_group_best", False),
            pick_score=details.get("pick_score", 0.0),
        )


@dataclass(frozen=True)
class FilterCriteria:
    rating: int = 0
    tag: str = ""
    date: str = ""


@dataclass(frozen=True)
class ImageBrowserState:
    images: list[str]
    selected: list[str]
    primary: str | None
    rating: int
    tags: list[str]
    label: str | None
    has_selection: bool
    progress_current: int
    progress_total: int
    can_undo: bool = False
    can_redo: bool = False
    filtered_images: list[str] = field(default_factory=list)
    filtered_count: int = 0
    predicted_labels: dict[str, str] = field(default_factory=dict)  # filename -> 'keep'/'trash'/''
    predicted_probabilities: dict[str, float] = field(default_factory=dict)  # filename -> probability (0.0-1.0)
    auto_assigned_paths: set = field(default_factory=set)  # paths that were auto-labeled (vs manual)
    detected_objects: dict[str, list[str]] = field(default_factory=dict)  # filename -> list of detected objects
    # Detection backend/device info (for UI display)
    detection_backend: str = "unknown"
    detection_device: str = "unknown"
    # Grouping data
    group_info: dict[str, dict] = field(default_factory=dict)  # filename -> {group_id, group_size, is_group_best, pick_score}
