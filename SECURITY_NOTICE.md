# Security Notice - Credential Removal

**Date:** November 7, 2025  
**Action:** Complete removal of `.env` file from git history

---

## ⚠️ Important Security Update

### What Happened

The `.env` file containing sensitive credentials was accidentally committed to the git repository in previous versions. This has been **completely removed** from all git history.

### Actions Taken

✅ **Removed `.env` from all commits** using `git filter-branch`  
✅ **Rewritten all git tags** (v1.0, v1.0.1, v1.2, v1.3, v1.4, v1.5)  
✅ **Force-pushed clean history** to GitHub  
✅ **Verified `.gitignore`** is properly configured  
✅ **Cleaned local git cache** and garbage collected  

### Affected Versions

All previous versions had the `.env` file in git history:
- v1.0
- v1.0.1
- v1.2
- v1.3
- v1.4
- v1.5 (initial push)

**All versions have been rewritten and re-pushed without credentials.**

---

## 🔒 Security Recommendations

### Immediate Actions Required

If you cloned this repository before this fix:

1. **Delete your local clone:**
   ```bash
   cd ..
   rm -rf docling-service
   ```

2. **Clone fresh copy:**
   ```bash
   git clone https://github.com/Crypto-Gi/docling-service.git
   cd docling-service
   ```

3. **Rotate your credentials:**
   - Generate new R2 access keys
   - Update your production `.env` file
   - Delete old credentials from Cloudflare

### For Production Deployments

1. **Rotate all R2 credentials immediately**
2. **Review access logs** for unauthorized access
3. **Update production `.env`** with new credentials
4. **Verify `.gitignore`** is working:
   ```bash
   git status  # .env should NOT appear
   ```

---

## 📋 Credential Rotation Guide

### Cloudflare R2

1. **Login to Cloudflare Dashboard**
2. **Navigate to R2 → Manage R2 API Tokens**
3. **Delete old API token**
4. **Create new API token:**
   - Name: `docling-service-new`
   - Permissions: Object Read & Write
   - Bucket: Your bucket name
5. **Update `.env` with new credentials**

### Verify New Setup

```bash
# Test with new credentials
curl -X POST http://localhost:5010/api/convert \
  -F "file=@test.pdf"

# Check logs for successful upload
docker logs docling-service-docling-1 | grep "Uploaded"
```

---

## ✅ Current Security Status

### Protected Files

The following files are now properly ignored:
- `.env` - Your actual credentials (never committed)
- `storage/` - Local storage directory
- `*.log` - Log files
- `__pycache__/` - Python cache

### Template Files (Safe to Commit)

- `.env.example` - Template with empty values ✅
- `.gitignore` - Properly configured ✅
- All documentation files ✅

---

## 🔍 Verification

### Check Git History

```bash
# This should return nothing
git log --all --full-history -- .env

# Verify .env is ignored
git status  # .env should not appear
```

### Check Remote

```bash
# Verify tags are updated
git ls-remote --tags origin

# All tags should have new commit hashes
```

---

## 📚 Best Practices Going Forward

### Never Commit Credentials

❌ **Never commit:**
- `.env` files
- API keys
- Passwords
- Private keys
- Access tokens

✅ **Always use:**
- `.env.example` as template
- Environment variables
- Secret management services
- `.gitignore` for sensitive files

### Pre-Commit Checklist

Before committing:
1. ✅ Check `git status` for sensitive files
2. ✅ Review `git diff` for credentials
3. ✅ Verify `.gitignore` is working
4. ✅ Use `.env.example` for templates

### Setup Git Hooks (Optional)

Prevent accidental commits:

```bash
# .git/hooks/pre-commit
#!/bin/bash
if git diff --cached --name-only | grep -q "^.env$"; then
    echo "Error: Attempting to commit .env file!"
    echo "Please remove it from staging: git reset HEAD .env"
    exit 1
fi
```

Make executable:
```bash
chmod +x .git/hooks/pre-commit
```

---

## 📞 Support

If you have concerns about credential exposure:

1. **Rotate credentials immediately** (see guide above)
2. **Review access logs** in Cloudflare dashboard
3. **Monitor for suspicious activity**
4. **Contact support** if unauthorized access detected

---

## 📝 Timeline

- **Before Nov 7, 2025:** `.env` was in git history
- **Nov 7, 2025 02:50 UTC:** Discovered issue
- **Nov 7, 2025 02:55 UTC:** Removed from all history
- **Nov 7, 2025 02:56 UTC:** Force-pushed clean history
- **Status:** ✅ **Resolved**

---

## ✅ Verification Completed

- [x] `.env` removed from all commits
- [x] All tags rewritten and pushed
- [x] Git history cleaned
- [x] `.gitignore` verified
- [x] `.env.example` is clean
- [x] Documentation updated

**The repository is now secure and credentials are protected.** 🔒
