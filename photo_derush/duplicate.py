def cluster_duplicates(image_paths):
    """
    Minimal implementation: each image is its own group.
    Returns:
        clusters: List[List[str]] - each sublist is a group of image paths
        image_hashes: List[str] - dummy hash for each image
    """
    clusters = [[img] for img in image_paths]
    image_hashes = [f"dummyhash_{i}" for i in range(len(image_paths))]
    return clusters, image_hashes
