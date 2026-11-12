# SBVS website - Phase 1 (put the current app online)

This folder is a copy of the SBVS 2.0 app prepared for **Streamlit Community Cloud**
(free hosting). The class submission in `sbvs-2/` is untouched.

What changed compared with `sbvs-2/`:
- `requirements.txt` uses `opencv-python-headless` (cloud servers have no screen).
- `.streamlit/config.toml` raises the upload limit to 300 MB and sets a dark/orange theme.
- Only the app's `.py` files and sample data are included (no notebooks, no legacy files).

## What it can analyze

Fixed-camera footage only (phone on a tripod, hoop always in the same spot). It will
NOT give useful results on TV broadcasts (camera pans, cuts, replays). That is Phase 2.

## Deploy steps (about 20 minutes, done from your own accounts)

1. Unzip `sbvs-web-deploy.zip` on your laptop.
2. On GitHub, create a new repository (for example `sbvs-web`). Tick nothing else.
3. On the empty repo page click **uploading an existing file**. Drag in everything from the
   unzipped folder (files and folders), then **Commit changes**.
   - The `.streamlit` folder is hidden. Mac: press Cmd+Shift+. in Finder to show it.
     Windows: View > Show > Hidden items. If it still doesn't upload, use
     **Add file > Create new file**, name it `.streamlit/config.toml`, and paste its contents.
   - Check the repo shows `demo/duke_alabama_ball_tracking.mp4` and `broadcast/yolo11n.pt`.
4. Go to https://share.streamlit.io, sign in with GitHub, click **Create app** >
   **Deploy a public app from GitHub**.
5. Repository: your repo, branch `main`, main file `app.py`.
   Open **Advanced settings** and pick **Python 3.12**. Click **Deploy**.
6. The first build takes about 5 to 10 minutes (PyTorch is big). You get a URL like
   `https://sbvs-web.streamlit.app`. Open the **Demo** page to check the video plays.

If the build fails, open **Manage app** (bottom right) to see the log, and send it to Claude.

## Known limits of the free host

- At most about 2.7 GB of memory (if the Broadcast page uses too much, the app restarts;
  it shows an error message for smaller failures) and 2 shared CPU cores: keep clips short (a few minutes).
- Files the app saves (`data/shots.csv`, uploaded videos) are wiped when the app restarts,
  and every visitor shares the same saved data. Download your CSVs after each session.
- The app sleeps after 12 hours without visitors; opening the URL wakes it up.
- The Broadcast page needs PyTorch + EasyOCR (large). If the free host runs out of memory,
  run it on your laptop (`streamlit run app.py`) until Phase 3 moves it to a real backend.
