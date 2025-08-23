Plan:
1. [x] Set up minimal project structure
2. [x] Implement GUI skeleton
3. [x] Add directory selection and file listing
4. [x] Move long tasks off GUI thread
5. [x] Add tests
6. [x] Add image preview on selection
7. [x] Add file info panel for selected image
8. [x] Remember last selected directory
9. [x] Display EXIF metadata for selected image
10. [x] Display images as a grid (Lightroom-style UI)
11. [x] Add thumbnail/miniature caching for faster preview
12. [x] Add file type filter (e.g., .jpg/.png only)
13. [x] Add open-in-system-viewer for selected image
14. [x] Add image rating/tagging support
15. [x] Add search/filter by filename or EXIF
16. [ ] Add responsive grid layout with selection highlight (Lightroom-style polish)
17. [ ] Add info/metadata side panel with editable tags
18. [ ] Add quick filter bar (by rating/tag/date)
19. [ ] Add export/copy/move selected images
20. [ ] Add batch metadata editing
21. [ ] Add fullscreen/compare mode for selected images
22. [ ] Add settings/preferences dialog

Steps Executed:
- 2025-08-23: Set up minimal project structure
- 2025-08-23: Implemented GUI skeleton
- 2025-08-23: Added directory selection and file listing
- 2025-08-23: Moved long tasks off GUI thread
- 2025-08-23: Added tests
- 2025-08-23: Ran app and verified basic functionality
- 2025-08-23: Planning and starting image preview feature
- 2025-08-23: Implemented and verified image preview feature
- 2025-08-23: Planning and starting file info panel feature
- 2025-08-23: Implemented and verified file info panel feature
- 2025-08-23: Implemented and verified last directory persistence
- 2025-08-23: Planning and starting EXIF display feature
- 2025-08-23: Implemented and verified EXIF display feature
- 2025-08-23: Assessed thumbnail/miniature caching (not present)
- 2025-08-23: Planning next features for usability and performance
- 2025-08-23: Planned detailed next steps for project, including caching, filters, batch actions, tagging, search, export, and settings
- 2025-08-23: Planned Lightroom-style grid UI for image display
- 2025-08-23: Implemented Lightroom-style grid UI in main app using existing modular components
- 2025-08-23: Removed try/except from import of Resampling/LANCZOS in app.py, used importlib fallback logic
- 2025-08-23: Updated plan for Lightroom-style UI, added more Lightroom-inspired features
- 2025-08-23: Refactored app to idiomatic MVVM (model, view, viewmodel separation)
- 2025-08-23: Made image loading and thumbnail generation fully asynchronous and incremental
- 2025-08-23: Fixed grid flicker and ensured thumbnails remain visible
- 2025-08-23: Fixed progress bar to show and update as images load, and hide when done
- 2025-08-23: Implemented and verified thumbnail/miniature caching for faster preview
- 2025-08-23: Implemented and verified file type filter (e.g., .jpg/.png only)
- 2025-08-23: Implemented and verified open-in-system-viewer for selected image
- 2025-08-23: Refactored model.py for robust logging and input validation
- 2025-08-23: Added comprehensive unit tests for ImageModel (input validation, error handling, persistence)
- 2025-08-23: All tests pass for ImageModel, confirming robust error handling and correct behavior
- 2025-08-23: Implemented and tested search/filter by filename and EXIF in ImageModel
- 2025-08-23: All tests pass for search/filter features, including edge cases

Progress: 85%
Next step: Add responsive grid layout with selection highlight (Lightroom-style polish)
