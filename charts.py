# charts.py
# SBVS 2.0 - Matplotlib charts for the stats dashboard (owner: Jessica)
#
# Interface (see docs/INTERFACES.md):
#   fg_by_player_chart(player_table, team_fg) -> Matplotlib figure
#   attempts_vs_makes_chart(player_table)     -> Matplotlib figure
#   running_fg_chart(shots_df, player_table)  -> Matplotlib figure (idea from SBVS 1.0)
#
# player_table is the DataFrame returned by stats.player_stats().
# Each function builds a figure with plt.subplots() and returns it.
# The app shows it with st.pyplot(fig); fig.savefig('name.png') saves it for the report.

import numpy as np
import matplotlib.pyplot as plt

# Colors (kept the same in every chart so a color always means the same thing)
ATTEMPTS_COLOR = '#2a78d6'   # blue
MAKES_COLOR = '#eb6834'      # orange
TEAM_LINE_COLOR = '#52514e'  # dark gray for the team average line
TEXT_COLOR = '#0b0b0b'

# Line colors for running_fg_chart, used in this fixed order: the 1st player in the
# table always gets blue, the 2nd orange, and so on. These 8 colors were picked so
# neighbors stay different for colorblind readers. With more than 8 players the
# colors start again from the top (a class team is never that big).
PLAYER_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
                 '#e87ba4', '#008300', '#6250d6', '#e34948']


def player_labels(player_table):
    """Make x-axis labels like '#23 Arham' (jersey + first name)."""
    labels = []
    for i in range(len(player_table)):
        row = player_table.iloc[i]
        first_name = row['name'].split(' ')[0]
        labels.append(f"#{row['jersey']} {first_name}")
    return labels


def style_axes(ax):
    """Light grid behind the bars and no top/right border lines."""
    # Light gray horizontal grid lines make the values easier to read
    ax.grid(axis='y', color='#dddddd', linewidth=0.8)
    # Draw the grid behind the bars instead of on top of them
    ax.set_axisbelow(True)
    # "Spines" are the 4 border lines of the plot; hide the top and right ones
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def fg_by_player_chart(player_table, team_fg):
    """Bar chart of each player's FG%, with a dashed line for the team FG%.
    Players with 0 attempts have no bar and are marked 'no shots'.
    """
    labels = player_labels(player_table)
    x = np.arange(len(labels))
    # NaN FG% (0 attempts) is drawn as a bar of height 0
    heights = player_table['fg_pct'].fillna(0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, heights, width=0.6, color=ATTEMPTS_COLOR, label='Player FG%')

    # Write the value above each bar so the chart can be read without the axis.
    # ax.text(x, y, words) puts text at that spot; ha/va center it on the bar.
    for i in range(len(player_table)):
        row = player_table.iloc[i]
        if row['attempts'] > 0:
            text = f"{row['fg_pct']:.1f}%\n({row['makes']}/{row['attempts']})"
            ax.text(x[i], row['fg_pct'] + 2, text, ha='center', va='bottom',
                    fontsize=9, color=TEXT_COLOR)
        else:
            ax.text(x[i], 2, 'no shots', ha='center', va='bottom',
                    fontsize=9, color=TEAM_LINE_COLOR)

    # Team FG% line (skipped if the team has no counted shots yet).
    # A flat line is drawn with plt.plot using the same y value at every x,
    # like the scrap-limit line in the Ch6 slides.
    if not np.isnan(team_fg):
        line_x = [-0.5, len(labels) - 0.5]
        line_y = [team_fg, team_fg]
        ax.plot(line_x, line_y, color=TEAM_LINE_COLOR, linestyle='--', linewidth=1.5,
                label=f"Team FG% ({team_fg:.1f}%)")
        ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.legend(loc='upper right', frameon=False)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel('Field goal % (FG%)')
    ax.set_title('Field goal percentage by player')
    style_axes(ax)
    # tight_layout() shrinks the margins so labels and title are not cut off
    fig.tight_layout()
    return fig


def attempts_vs_makes_chart(player_table):
    """Side-by-side bars: shot attempts and makes for each player."""
    labels = player_labels(player_table)
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, player_table['attempts'], width=width - 0.02,
           color=ATTEMPTS_COLOR, label='Attempts')
    ax.bar(x + width / 2, player_table['makes'], width=width - 0.02,
           color=MAKES_COLOR, label='Makes')

    # Count above each bar. ax.text(x, y, words) puts text at that spot.
    for i in range(len(player_table)):
        row = player_table.iloc[i]
        ax.text(x[i] - width / 2, row['attempts'], str(row['attempts']),
                ha='center', va='bottom', fontsize=9, color=TEXT_COLOR)
        ax.text(x[i] + width / 2, row['makes'], str(row['makes']),
                ha='center', va='bottom', fontsize=9, color=TEXT_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    # Leave room above the tallest bar for its label
    top = player_table['attempts'].max()
    if top == 0:
        top = 1
    ax.set_ylim(0, top * 1.2)
    ax.set_ylabel('Number of shots')
    ax.set_title('Shot attempts vs makes by player')
    ax.legend(loc='upper right', frameon=False)
    style_axes(ax)
    # tight_layout() shrinks the margins so labels and title are not cut off
    fig.tight_layout()
    return fig


def running_fg_values(shots_df, player_id):
    """A player's FG% after each of their shots, in time order.
    Shots are sorted by session, then by time inside the session.
    Only 'made' and 'missed' count. Returns two lists:
    shot numbers [1, 2, 3, ...] and the running FG% after each shot.
    Example: made, missed, made -> [1, 2, 3] and [100.0, 50.0, 66.7]
    """
    mask = (shots_df['result'] == 'made') | (shots_df['result'] == 'missed')
    counted = shots_df[mask]
    mine = counted[counted['player_id'] == player_id]
    # Session ids start with the date (YYYY-MM-DD), so sorting the text sorts by date
    mine = mine.sort_values(['session_id', 'time_s'])

    shot_numbers = []
    running_pct = []
    makes = 0
    attempts = 0
    for result in mine['result']:
        attempts = attempts + 1
        if result == 'made':
            makes = makes + 1
        shot_numbers.append(attempts)
        running_pct.append(round(makes / attempts * 100, 1))
    return shot_numbers, running_pct


def running_fg_chart(shots_df, player_table):
    """Line chart of each player's running FG% (idea from SBVS 1.0).
    x axis = the player's shot number (1st shot, 2nd shot, ...),
    y axis = their FG% counting all shots up to that one.
    Players with no counted shots get no line.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    longest = 0

    for i in range(len(player_table)):
        row = player_table.iloc[i]
        shot_numbers, running_pct = running_fg_values(shots_df, row['player_id'])
        if len(shot_numbers) == 0:
            continue
        # % gives the remainder, so after the 8th color it starts again at the 1st
        color = PLAYER_COLORS[i % len(PLAYER_COLORS)]
        ax.plot(shot_numbers, running_pct, color=color, linewidth=2,
                marker='o', markersize=5, label=f"#{row['jersey']} {row['name']}")
        if len(shot_numbers) > longest:
            longest = len(shot_numbers)

    if longest == 0:
        # Nothing to draw yet: say so on the empty chart
        ax.text(0.5, 50, 'No counted shots yet', ha='center', va='center',
                fontsize=11, color=TEAM_LINE_COLOR)
        ax.set_xlim(0, 1)
    else:
        ax.set_xticks(np.arange(1, longest + 1))
        ax.set_xlim(0.5, longest + 0.5)
        ax.legend(loc='lower right', frameon=False)

    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Player's shot number (sessions in date order)")
    ax.set_ylabel('Running FG% (%)')
    ax.set_title('Running field goal percentage by player')
    style_axes(ax)
    # tight_layout() shrinks the margins so labels and title are not cut off
    fig.tight_layout()
    return fig
