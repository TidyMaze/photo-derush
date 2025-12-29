import logging
from dataclasses import dataclass

from .domain import FilterCriteria


@dataclass
class FilteringService:
    """Service encapsulating filtering logic so ImageModel stays lean.

    Methods expect ImageModel-like object with:
      - get_image_files()
      - get_image_path(filename)
      - get_rating(path)
      - get_tags(path)
      - load_exif(path)
    """

    def filter_by_rating_tag_date(self, image_model, criteria: FilterCriteria) -> list[str]:
        rating = criteria.rating or 0
        tag = (criteria.tag or "").strip().lower()
        date = (criteria.date or "").strip()
        files = image_model.get_image_files()
        result: list[str] = []
        for f in files:
            path = image_model.get_image_path(f)
            file_rating = image_model.get_rating(path)
            if rating and int(file_rating) < int(rating):
                continue
            if tag:
                tags = [t.lower() for t in image_model.get_tags(path)]
                if tag not in tags:
                    continue
            if date:
                exif = image_model.load_exif(path)
                exif_date = exif.get("DateTimeOriginal") or exif.get("DateTime") or ""
                if not str(exif_date).startswith(date):
                    continue
            result.append(f)
        logging.debug("FilteringService result: %s", result)
        return result
