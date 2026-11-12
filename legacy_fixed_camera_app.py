# app.py
# SBVS 2.0 - the Streamlit app (owner: Jonathan)
# Drafted with help from Claude (an AI assistant); Jonathan reviews, runs,
# and commits it himself.
#
# Run it with:   streamlit run app.py   (or double-click run_sbvs.bat / run_sbvs.command)
#
# How Streamlit works, in one paragraph:
#   Streamlit runs this whole file from top to bottom every time the user
#   clicks or types something. Normal variables are forgotten between runs,
#   so anything we must remember (the video, the hoop, the shots, the roster)
#   is kept in st.session_state, which works like a dictionary that survives
#   between runs. The keys we use are fixed in docs/INTERFACES.md:
#   'video_path', 'hoop', 'shots', 'players'.
#
# Screens (picked with the radio buttons in the sidebar):
#   Upload -> Hoop -> Analyze -> Review -> Players -> Dashboard
#
# This is the only file that imports streamlit. All the real work is done by
# the team's modules: tracker.py (Arham), shots.py (Kevin),
# players.py / stats.py / charts.py (Jessica), pipeline.py (Jonathan).

import os

import numpy as np
import pandas as pd
import streamlit as st

import settings
import tracker
import shots
import players
import stats
import charts
import pipeline


SCREENS = ['Upload', 'Hoop', 'Analyze', 'Review', 'Players', 'Dashboard']
RESULTS = ['made', 'missed', 'unknown']

RECORDING_TIPS = """
**Recording tips (SBVS 2.0 was built for videos like this):**
- Camera on a tripod or fixed, not moving
- The whole hoop visible for the whole video
- Side view or 45 degree view of the shooter and hoop
- 720p or higher, 30 frames per second or more

Videos recorded differently may not work well.
"""

# Keep this note until the team has measured accuracy on real labeled clips.
ACCURACY_NOTE = ('Accuracy has not been measured on real clips yet. '
                 'Check every result on the Review screen.')


# ---------------------------------------------------------------------------
# Small helper functions (no Streamlit inside, easy to test in a notebook)
# ---------------------------------------------------------------------------

def make_folders():
    """Create the data folders the first time the app runs."""
    folder_list = [settings.DATA_FOLDER, settings.VIDEO_FOLDER,
                   settings.POSITIONS_FOLDER]
    for folder in folder_list:
        if not os.path.exists(folder):
            os.makedirs(folder)


def player_choices(player_list):
    """Return two lists for the player dropdown: the labels people see,
    and the player_id that goes with each label. Index 0 is 'Unassigned' (id 0).
    """
    labels = ['Unassigned']
    ids = [0]
    for player in player_list:
        labels.append(str(player))     # for example '#23 Arham Ahmad'
        ids.append(player.player_id)
    return labels, ids


def fg_text(value):
    """FG% as text for the screen: '54.5%', or '-' when there were no attempts."""
    if np.isnan(value):
        return '-'
    return f"{value:.1f}%"


def shot_counts(shots_df):
    """Count made, missed and unknown shots. Returns a dictionary."""
    made = int((shots_df['result'] == 'made').sum())
    missed = int((shots_df['result'] == 'missed').sum())
    unknown = int((shots_df['result'] == 'unknown').sum())
    return {'made': made, 'missed': missed, 'unknown': unknown}


def stats_table_for_screen(player_table):
    """Copy of the player stats table with FG% as text ('-' for no attempts)."""
    shown = player_table.copy()
    fg_list = []
    for value in player_table['fg_pct']:
        fg_list.append(fg_text(value))
    shown['fg_pct'] = fg_list
    return shown


def clear_review_choices():
    """Forget the made/missed and player choices from an older shot list,
    so the Review screen starts from the newly loaded shots.
    """
    old_keys = []
    # st.session_state works like a dictionary; .keys() lists what is saved
    for key in st.session_state.keys():
        if key.startswith('result_') or key.startswith('player_'):
            old_keys.append(key)
    for key in old_keys:
        # Remove one saved widget value
        del st.session_state[key]


def start_session_state():
    """Give every session_state key a starting value on the first run."""
    # st.session_state remembers values between reruns of this file
    if 'video_path' not in st.session_state:
        st.session_state['video_path'] = None
    if 'hoop' not in st.session_state:
        st.session_state['hoop'] = None
    if 'shots' not in st.session_state:
        st.session_state['shots'] = None
    if 'players' not in st.session_state:
        st.session_state['players'] = players.load_players(settings.PLAYERS_FILE)


# ---------------------------------------------------------------------------
# Screens. Each screen is one function.
# ---------------------------------------------------------------------------

def upload_screen():
    # Big title at the top of the page
    st.header('1. Upload a video')
    # Markdown text: the recording tips (bold and a bullet list)
    st.markdown(RECORDING_TIPS)
    # Yellow warning box: accuracy is not measured yet
    st.warning(ACCURACY_NOTE)
    # Grey hint text about how to name videos
    st.caption('Name videos like 2026-11-04_practice_1.mp4 or 2026-11-05_game_2.mp4 '
               'so the date and session type are saved with the shots.')

    # A file picker that only accepts .mp4 and .mov files
    uploaded = st.file_uploader('Choose a basketball video', type=['mp4', 'mov'])

    if uploaded is not None:
        video_path = settings.VIDEO_FOLDER + '/' + uploaded.name
        # Streamlit reruns this file on every click, so only save the video
        # the first time we see it.
        if video_path != st.session_state['video_path']:
            # 'wb' = write bytes, because a video is not text
            video_file = open(video_path, 'wb')
            # getbuffer() gives the bytes of the uploaded file
            video_file.write(uploaded.getbuffer())
            video_file.close()
            st.session_state['video_path'] = video_path
            # A new video needs a new hoop and new shots
            st.session_state['hoop'] = None
            st.session_state['shots'] = None
            clear_review_choices()

    video_path = st.session_state['video_path']
    if video_path is None:
        # Blue information box
        st.info('No video uploaded yet.')
        return

    # Green success box with the saved video's path
    st.success(f"Current video: {video_path}")
    name_info = pipeline.parse_video_name(video_path)
    if name_info is None:
        # Yellow warning box: the name doesn't follow the pattern
        st.warning('This file name does not follow the naming pattern. '
                   'The video still works, but no date or session type is saved.')
    else:
        # Normal text line with the date, type and clip number from the name
        st.write(f"Date: {name_info['date']}   Type: {name_info['kind']}   "
                 f"Clip: {name_info['number']}")
    # Video player with a play button
    st.video(video_path)


def hoop_screen():
    # Big title at the top of the page
    st.header('2. Mark the hoop')

    video_path = st.session_state['video_path']
    if video_path is None:
        # Blue information box
        st.info('Upload a video first.')
        return

    info = tracker.video_info(video_path)
    frame = tracker.read_frame(video_path, 0)
    if frame is None:
        # Red error box
        st.error('Could not read the first frame of this video.')
        return

    max_x = int(info['work_width'])
    max_y = int(info['work_height'])

    # Start the sliders at the saved hoop, or at a box in the middle of the frame
    if st.session_state['hoop'] is not None:
        box = st.session_state['hoop'].to_dict()
    else:
        box = {'x1': max_x // 2 - 40, 'y1': max_y // 3 - 15,
               'x2': max_x // 2 + 40, 'y2': max_y // 3 + 15}

    # Normal text line with instructions
    st.write('Move the sliders until the box fits tightly around the rim. '
             'The middle line of the box is the rim line used for made shots.')
    # Two side-by-side columns, so the x sliders and y sliders sit together
    left, right = st.columns(2)
    # Slider in the left column: label, smallest value, biggest value, starting value
    x1 = left.slider('Left edge (x1)', 0, max_x, int(box['x1']))
    # Slider in the left column
    x2 = left.slider('Right edge (x2)', 0, max_x, int(box['x2']))
    # Slider in the right column
    y1 = right.slider('Top edge (y1)', 0, max_y, int(box['y1']))
    # Slider in the right column
    y2 = right.slider('Bottom edge (y2)', 0, max_y, int(box['y2']))

    if x1 >= x2 or y1 >= y2:
        # Red error box
        st.error('The left edge must be left of the right edge, and the top edge '
                 'must be above the bottom edge.')
        return

    hoop = tracker.Hoop(x1, y1, x2, y2)
    preview = tracker.draw_hoop(frame, hoop)
    # Show the picture. OpenCV frames are BGR, so we tell Streamlit that.
    st.image(preview, channels='BGR', caption='First frame with the hoop box')
    # Normal text line with the rim size
    st.write(f"Rim width: {hoop.width():.0f} px. Expected ball radius: "
             f"{hoop.ball_radius():.1f} px.")

    # A button; it returns True only on the run right after it is clicked
    if st.button('Save hoop'):
        st.session_state['hoop'] = hoop
        # Green success box
        st.success('Hoop saved. Go to Analyze.')


def analyze_screen():
    # Big title at the top of the page
    st.header('3. Analyze the video')

    video_path = st.session_state['video_path']
    hoop = st.session_state['hoop']
    if video_path is None or hoop is None:
        # Blue information box
        st.info('Upload a video and save the hoop first.')
        return

    session_id = pipeline.session_id_from_name(video_path)
    # Normal text line with the video path
    st.write(f"Video: {video_path}")
    # Normal text line with the session id
    st.write(f"Session: {session_id}")
    # Grey hint text
    st.caption('Tracking looks at every frame, so a long video can take a few minutes. '
               'Analyzing the same video again replaces its old shots, '
               'including any changes made on the Review screen.')

    # A button; it returns True only on the run right after it is clicked
    if st.button('Analyze'):
        # A progress bar from 0.0 to 1.0; the tracker moves it forward
        progress_bar = st.progress(0.0)
        shots_df = pipeline.analyze_video(video_path, hoop, session_id, progress_bar)
        shots.save_session_shots(shots_df, settings.SHOTS_FILE)
        st.session_state['shots'] = shots_df
        clear_review_choices()
        # Green success box
        st.success(f"Found {len(shots_df)} shots. Saved to {settings.SHOTS_FILE}.")

    # The quick button only works after the video was tracked once
    if os.path.exists(pipeline.positions_path(session_id)):
        # Grey hint text
        st.caption('Re-run shot rules uses the saved ball positions, so it skips '
                   'tracking and takes about a second. Use it after changing shots.py.')
        # A second button for the quick re-run
        if st.button('Re-run shot rules'):
            shots_df = pipeline.rerun_shot_rules(session_id, video_path, hoop)
            shots.save_session_shots(shots_df, settings.SHOTS_FILE)
            st.session_state['shots'] = shots_df
            clear_review_choices()
            # Green success box
            st.success(f"Shot rules re-run: {len(shots_df)} shots. "
                       f"Saved to {settings.SHOTS_FILE}.")

    shots_df = st.session_state['shots']
    if shots_df is not None:
        counts = shot_counts(shots_df)
        # Three side-by-side columns for the three numbers
        col1, col2, col3 = st.columns(3)
        # A big number with a label, in the first column
        col1.metric('Made', counts['made'])
        # A big number with a label, in the second column
        col2.metric('Missed', counts['missed'])
        # A big number with a label, in the third column
        col3.metric('Needs review', counts['unknown'])
        # A scrollable table of the shots
        st.dataframe(shots_df[['shot_id', 'time_s', 'result', 'player_id']],
                     hide_index=True)


def review_screen():
    # Big title at the top of the page
    st.header('4. Review shots and assign players')

    # If nothing was analyzed in this visit, let the user open a saved session
    all_shots = shots.load_all_shots(settings.SHOTS_FILE)
    if len(all_shots) > 0:
        session_list = list(all_shots['session_id'].unique())
        # Dropdown of saved sessions
        chosen = st.selectbox('Saved session', session_list)
        # A button; it returns True only on the run right after it is clicked
        if st.button('Open this session'):
            one_session = all_shots[all_shots['session_id'] == chosen]
            st.session_state['shots'] = one_session.reset_index(drop=True)
            clear_review_choices()

    shots_df = st.session_state['shots']
    if shots_df is None or len(shots_df) == 0:
        # Blue information box
        st.info('No shots to review yet. Analyze a video or open a saved session.')
        return

    shots_df = shots_df.reset_index(drop=True)
    session_id = shots_df['session_id'].iloc[0]
    labels, ids = player_choices(st.session_state['players'])

    # Normal text line with instructions
    st.write(f"Session {session_id}: {len(shots_df)} shots. Change anything that is "
             "wrong, then press Save changes.")
    if len(ids) == 1:
        # Blue information box
        st.info('The roster is empty. Add players on the Players screen.')

    new_results = []
    new_player_ids = []
    for i in range(len(shots_df)):
        row = shots_df.iloc[i]
        shot_id = int(row['shot_id'])

        # Pick the current choices so the widgets start at the saved values
        result_index = 0
        if row['result'] in RESULTS:
            result_index = RESULTS.index(row['result'])
        player_index = 0
        if int(row['player_id']) in ids:
            player_index = ids.index(int(row['player_id']))

        # Three columns: shot info, made/missed buttons, player dropdown
        col1, col2, col3 = st.columns([1, 2, 2])
        # Normal text in the first column
        col1.write(f"Shot {shot_id} at {row['time_s']:.1f} s")
        # Round buttons for made / missed / unknown. Each key must be unique.
        result = col2.radio('Result', RESULTS, index=result_index, horizontal=True,
                            key='result_' + session_id + '_' + str(shot_id))
        # Dropdown of players. Each key must be unique.
        label = col3.selectbox('Player', labels, index=player_index,
                               key='player_' + session_id + '_' + str(shot_id))
        new_results.append(result)
        new_player_ids.append(ids[labels.index(label)])

    # A button; it returns True only on the run right after it is clicked
    if st.button('Save changes'):
        changed = 0
        for i in range(len(shots_df)):
            old_result = shots_df.loc[i, 'result']
            old_player = int(shots_df.loc[i, 'player_id'])
            if new_results[i] != old_result or new_player_ids[i] != old_player:
                shots_df.loc[i, 'result'] = new_results[i]
                shots_df.loc[i, 'player_id'] = new_player_ids[i]
                shots_df.loc[i, 'edited'] = True
                changed = changed + 1
        # made is 1 only for 'made' shots (same rule as shots.py)
        shots_df['made'] = np.where(shots_df['result'] == 'made', 1, 0)
        shots.save_session_shots(shots_df, settings.SHOTS_FILE)
        st.session_state['shots'] = shots_df
        # Green success box
        st.success(f"Saved. {changed} shots changed.")


def players_screen():
    # Big title at the top of the page
    st.header('5. Players')
    player_list = st.session_state['players']

    # Small heading
    st.subheader('Add a player')
    # Text box the user types the name into
    name_text = st.text_input('Name')
    # Text box the user types the jersey number into
    jersey_text = st.text_input('Jersey number (1 or 2 digits, for example 5, 23 or 00)')

    # A button; it returns True only on the run right after it is clicked
    if st.button('Add player'):
        name = players.clean_name(name_text)
        jersey = jersey_text.strip()
        jersey_taken = False
        for player in player_list:
            if player.jersey == jersey:
                jersey_taken = True

        if name == '':
            # Red error box
            st.error('Please type a name.')
        elif not players.is_valid_jersey(jersey):
            # Red error box
            st.error('The jersey number must be 1 or 2 digits.')
        elif jersey_taken:
            # Red error box
            st.error(f"Jersey #{jersey} is already on the roster.")
        else:
            new_id = players.next_player_id(player_list)
            player_list.append(players.Player(new_id, name, jersey))
            players.save_players(player_list, settings.PLAYERS_FILE)
            st.session_state['players'] = player_list
            # Green success box
            st.success(f"Added #{jersey} {name}.")

    # Small heading
    st.subheader('Roster')
    if len(player_list) == 0:
        # Blue information box
        st.info('No players yet.')
    else:
        # A scrollable table of the roster
        st.dataframe(players.players_to_dataframe(player_list), hide_index=True)


def dashboard_screen():
    # Big title at the top of the page
    st.header('6. Dashboard')

    # Tick box to look at Jessica's hand-made sample files instead of real data
    use_sample = st.checkbox('Show sample data instead (made-up shots, NOT real results)')
    if use_sample:
        shots_df = pd.read_csv(settings.SAMPLE_SHOTS_FILE)
        roster = players.load_players(settings.SAMPLE_PLAYERS_FILE)
        # Yellow warning box
        st.warning('SAMPLE DATA: these shots never happened. Do not use these numbers '
                   'in the report.')
    else:
        shots_df = shots.load_all_shots(settings.SHOTS_FILE)
        roster = st.session_state['players']

    if len(shots_df) == 0:
        # Blue information box
        st.info('No shots saved yet. Analyze a video first.')
        return

    session_list = ['All sessions'] + list(shots_df['session_id'].unique())
    # Dropdown to pick one session or all of them
    chosen = st.selectbox('Session', session_list)
    if chosen != 'All sessions':
        shots_df = shots_df[shots_df['session_id'] == chosen]

    players_df = players.players_to_dataframe(roster)
    player_table = stats.player_stats(shots_df, players_df)
    team = stats.team_stats(shots_df)

    # Small heading
    st.subheader('Team')
    # Four side-by-side columns for the team numbers
    col1, col2, col3, col4 = st.columns(4)
    # A big number with a label, in the first column
    col1.metric('Attempts', team['attempts'])
    # A big number with a label, in the second column
    col2.metric('Makes', team['makes'])
    # A big number with a label, in the third column
    col3.metric('Misses', team['misses'])
    # A big number with a label, in the fourth column
    col4.metric('FG%', fg_text(team['fg_pct']))
    # Grey hint text
    st.caption(f"{team['unknown']} shots need review (left out of FG%). "
               f"{team['unassigned']} counted shots are not assigned to a player.")

    # Small heading
    st.subheader('Players')
    if len(player_table) == 0:
        # Blue information box
        st.info('No players on the roster yet.')
        return
    # A scrollable table of the player stats
    st.dataframe(stats_table_for_screen(player_table), hide_index=True)

    # Jessica's FG% chart is a Matplotlib figure; st.pyplot shows a figure
    st.pyplot(charts.fg_by_player_chart(player_table, team['fg_pct']))
    # Jessica's attempts vs makes chart, shown the same way
    st.pyplot(charts.attempts_vs_makes_chart(player_table))

    # Button that lets the user save the shots table as a CSV file
    st.download_button('Download shots CSV', shots_df.to_csv(index=False),
                       file_name='sbvs_shots.csv', mime='text/csv')
    # Button that lets the user save the player stats table as a CSV file
    st.download_button('Download player stats CSV', player_table.to_csv(index=False),
                       file_name='sbvs_player_stats.csv', mime='text/csv')


def main():
    # Browser tab title and a wide page layout
    st.set_page_config(page_title='SBVS 2.0', layout='wide')
    make_folders()
    start_session_state()

    # Title in the sidebar
    st.sidebar.title('SBVS 2.0')
    # Radio buttons in the sidebar to pick a screen
    screen = st.sidebar.radio('Screen', SCREENS)

    if screen == 'Upload':
        upload_screen()
    elif screen == 'Hoop':
        hoop_screen()
    elif screen == 'Analyze':
        analyze_screen()
    elif screen == 'Review':
        review_screen()
    elif screen == 'Players':
        players_screen()
    else:
        dashboard_screen()


main()
