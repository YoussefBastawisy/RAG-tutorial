# Troubleshooting Common Issues

## Files Not Syncing

If files aren't syncing between devices, try these steps in order:

1. **Check connection** — make sure you're online. The desktop app shows a green dot in the system tray when connected.
2. **Pause and resume sync** — right-click the system tray icon → Pause Sync, wait 10 seconds, then Resume Sync. This often clears stuck transfers.
3. **Check storage limit** — if your account is over its storage limit, sync stops uploading new files (existing files still sync).
4. **Restart the app** — fully quit and relaunch. On Mac, use Cmd+Q rather than just closing the window.
5. **Check selective sync settings** — Settings → Sync → Selective Sync. The folder may be excluded from sync.

If none of these work, check our status page at status.cloudnest.example.com for ongoing incidents.

## Upload Failures

The most common causes of upload failures:

- **File over size limit** — Free and Personal plans have a 10 GB per-file limit; Pro and Business have a 100 GB per-file limit.
- **Unsupported characters in filename** — avoid these characters in filenames: `< > : " / \ | ? *`. Rename the file and try again.
- **Network timeout on large files** — large uploads on slow connections may time out. The desktop app handles this automatically; the web uploader does not. For files over 1 GB, use the desktop app.

## App Crashes on Launch

If the desktop app crashes immediately on launch:

1. **Mac:** Delete `~/Library/Application Support/CloudNest/cache` and relaunch.
2. **Windows:** Delete `%APPDATA%\CloudNest\cache` and relaunch.
3. **Linux:** Delete `~/.config/CloudNest/cache` and relaunch.

This clears the local cache without affecting your synced files. If the crash persists, reinstall the app from cloudnest.example.com/download. Reinstalling does not delete your files.

## Can't Find a File I Know I Uploaded

Check these locations in order:
1. **Trash** — files deleted in the last 30 days are recoverable. Click "Trash" in the sidebar.
2. **Version history** — if the file was overwritten, right-click → Version History to restore an earlier version.
3. **Search** — Pro and Business plans have full-text search. The file may have been moved to a folder you didn't expect.
4. **Activity log** — Settings → Activity. Shows who moved, deleted, or modified files in the last 90 days (Personal+) or 1 year (Business).

## Slow Performance

If the app is sluggish:
- Close and relaunch
- Check that you're on the latest version (Settings → About → Check for Updates)
- If you have more than 100,000 files synced, consider using selective sync to only sync folders you actively use
- On older machines (8 GB RAM or less), lower the "Concurrent uploads" setting in Preferences from 4 to 2
