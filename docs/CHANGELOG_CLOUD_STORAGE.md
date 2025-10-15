# Cloud Storage Implementation Changelog

## Version 1.3 - Bucket Directory Structure Update

### Changes Made

#### Updated Remote Key Path Structure
- **Before**: Images uploaded to `{task_id}/picture-1.png` in bucket root
- **After**: Images uploaded to `images/{task_id}/picture-1.png` to match local storage structure

#### Modified Files
1. **`app/main.py`** (Lines 215, 241):
   - Updated `remote_key` for PictureItem: `f"images/{task_id}/{filename}"`
   - Updated `remote_key` for TableItem: `f"images/{task_id}/{filename}"`

2. **`docs/CLOUD_STORAGE_SETUP.md`**:
   - Added bucket structure documentation
   - Added visual examples of directory layout
   - Added example Markdown output with full URLs

### Bucket Structure

All images are now organized under an `images/` directory in the bucket:

```
your-bucket/
└── images/
    ├── task_id_1/
    │   ├── picture-1.png
    │   ├── picture-2.png
    │   └── table-1.png
    └── task_id_2/
        ├── picture-1.png
        └── picture-2.png
```

### Benefits

1. **Consistency**: Matches local storage structure (`storage/images/{task_id}/`)
2. **Organization**: All conversion images grouped under `images/` directory
3. **Separation**: Easy to separate images from other bucket content
4. **Cleanup**: Easier to implement lifecycle policies on `images/` prefix
5. **Compatibility**: Works with existing `images/` directory in bucket

### URL Format

**Cloudflare R2 URLs**:
```
https://bucket-name.account-id.r2.cloudflarestorage.com/images/{task_id}/picture-1.png
```

**With Custom CDN**:
```
https://cdn.yourdomain.com/images/{task_id}/picture-1.png
```

### Migration Notes

If you have existing uploads without the `images/` prefix:
- Old format: `{task_id}/picture-1.png` (bucket root)
- New format: `images/{task_id}/picture-1.png` (under images/)

No migration needed - old and new formats can coexist. New uploads will use the `images/` prefix.

### Testing Checklist

- [ ] Upload PDF with images
- [ ] Verify bucket structure: `images/{task_id}/picture-N.png`
- [ ] Check Markdown contains correct URLs with `images/` prefix
- [ ] Verify images load in browser from generated URLs
- [ ] Test with custom CDN URL if configured

### Compatibility

- ✅ Backward compatible with existing code
- ✅ Works with existing `images/` directory in bucket
- ✅ No changes to API or UI
- ✅ No changes to local storage behavior
