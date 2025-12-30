# Security Audit for Public Repository

## ⚠️ CRITICAL ISSUES FOUND

### 1. `.env` File Tracked in Git
**Status:** ⚠️ **MUST FIX BEFORE MAKING PUBLIC**

- `.env` file is currently tracked in git (found via `git ls-files`)
- Contains `SECRET_KEY=dev-secret-key-change-in-production`
- While this appears to be a dev placeholder, tracking `.env` files is a security anti-pattern
- **Action Required:**
  ```bash
  git rm --cached .env
  git commit -m "Remove .env from git tracking"
  ```
- Verify `.env` is in `.gitignore` (already present ✅)

### 2. Personal File Paths Hardcoded
**Status:** ⚠️ **SHOULD FIX**

Found hardcoded personal paths in multiple files:
- `/Users/username/work/photo-derush` (29 files)
- `/Users/username/Pictures/photo-dataset` (scripts)
- Debug log paths: `/Users/username/work/photo-derush/.cursor/debug.log`

**Files with hardcoded paths:**
- `src/auto_label_manager.py` (many debug log writes)
- `src/view.py` (debug log writes)
- `src/viewmodel.py` (debug log writes)
- `scripts/*.py` (default paths to `~/Pictures/photo-dataset`)
- Various test/debug scripts

**Recommendation:**
- Replace hardcoded paths with environment variables or `~` expansion
- Remove or comment out debug logging to `.cursor/debug.log` before public release
- Use relative paths or configurable defaults

### 3. Personal Username in Code
**Status:** ⚠️ **SHOULD FIX**

Personal username appears in:
- File paths (as shown above)
- Debug logging statements
- Script default paths

**Recommendation:**
- Replace with generic paths or environment variables
- Use `os.path.expanduser("~")` instead of hardcoded `/Users/username/`

## ✅ SAFE - No Issues Found

### Credentials & Secrets
- ✅ No API keys found (AWS, GitHub, Stripe, etc.)
- ✅ No hardcoded passwords
- ✅ No database connection strings with credentials
- ✅ No SSH keys or certificates
- ✅ No bearer tokens or authorization headers

### Personal Information
- ✅ No email addresses found
- ✅ No phone numbers found
- ✅ No physical addresses found

### Network & Infrastructure
- ✅ No hardcoded IP addresses (only localhost/127.0.0.1 which is safe)
- ✅ No production database URLs
- ✅ SQLite database paths are local/relative (safe)

## 📋 PRE-PUBLICATION CHECKLIST

### Must Fix (Blockers)
- [ ] Remove `.env` from git tracking: `git rm --cached .env`
- [ ] Commit the removal
- [ ] Verify `.env` is in `.gitignore` (already done ✅)

### Should Fix (Best Practice)
- [ ] Remove or replace hardcoded paths in:
  - `src/auto_label_manager.py` (debug log paths)
  - `src/view.py` (debug log paths)
  - `src/viewmodel.py` (debug log paths)
  - Scripts with `~/Pictures/photo-dataset` defaults
- [ ] Replace personal username references with generic paths
- [ ] Consider removing debug logging to `.cursor/debug.log` or making it optional

### Optional (Nice to Have)
- [ ] Create `.env.example` template file
- [ ] Review git history for any previously committed secrets:
  ```bash
  git log --all --full-history -- .env
  ```

## 🔍 FILES TO REVIEW

### High Priority
1. `.env` - Remove from git tracking
2. `src/auto_label_manager.py` - Remove hardcoded debug log paths
3. `src/view.py` - Remove hardcoded debug log paths
4. `src/viewmodel.py` - Remove hardcoded debug log paths

### Medium Priority
5. `scripts/*.py` - Replace hardcoded `~/Pictures/photo-dataset` with configurable defaults
6. Any test files with hardcoded paths

## 📝 NOTES

- The `SECRET_KEY` in `.env` appears to be a dev placeholder (`dev-secret-key-change-in-production`)
- However, if this repo was ever public, the key would be exposed in git history
- All hardcoded paths are development/debug paths, not production secrets
- The personal username in paths is a privacy concern but not a security risk

## ✅ AFTER FIXES

Once the above issues are addressed, the repository should be safe to make public.

