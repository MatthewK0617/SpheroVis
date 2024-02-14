# Import necessary libraries
import rosbag
import numpy as np
from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
import argparse  # Import argparse for command-line parsing


# Function to convert ROS messages to pandas DataFrame
def ros_messages_to_dataframe(bag_path, point_cloud=True, tol_s=0.1, tol_t=0.3):
    # Initialize an empty list to store the data
    data = []

    # Open the ROS bag file
    bag = rosbag.Bag(bag_path)
    if point_cloud:
        topics = ["/natnet_ros/pointcloud"]
    else:
        topics = []
        for key in bag.get_type_and_topic_info().topics.keys():
            if key.split("/")[-1] == "pose":
                topics.append(key)

    # Iterate over each message in the specified topic
    for i, msg in enumerate(bag.read_messages(topics=topics)):
        # Convert the ROS time to a datetime object
        timestamp = datetime.fromtimestamp(msg.timestamp.to_sec())
        # Extract the data fields you're interested in (e.g., 'state' and 'control')
        # This assumes the message has attributes 'state' and 'control'
        if point_cloud:
            data.append(
                {
                    "timestamp": timestamp,
                    "markers": len(msg.message.points),
                    "states": np.array(
                        [
                            msg.message.points[i].__getstate__()
                            for i in range(len(msg.message.points))
                        ]
                    ),
                }
            )
        else:
            data.append(
                {
                    "timestamp": timestamp,
                    "name": msg.topic.split("/")[-2],
                    "type": "state",
                    "val": np.array(msg.message.pose.position.__getstate__()),
                }
            )
        # elif msg.topic == "/control/final":
        #     data.append({'timestamp':timestamp, 'type':'control', 'val':np.array(msg.message.control.__getstate__())})
        # data.append({'timestamp': timestamp, 'state': msg.state, 'control': msg.control})

    bag.close()
    df = pd.DataFrame(data)

    if point_cloud:
        data = []

        # Initial Localization
        Xnext = position_init(df.states[0], tol_s=tol_s)
        n_ag = Xnext.shape[0]
        data.append(
            {
                "timestamp": df.timestamp[0],
                "type": "state",
                "agents_total": n_ag,
                "agents_seen": np.ones(n_ag, dtype=int),
                "n_agents_seen": n_ag,
                "val": Xnext,
                "norms_trav": np.zeros(n_ag),
            }
        )  # initial, assumes all agents seen at first

        # Time-step Matching
        for i in range(1, len(df)):
            Xnext, norms_trav, ag_seen = match_position(
                df.states[i], Xnext, tol_t=tol_t
            )
            data.append(
                {
                    "timestamp": df.timestamp[i],
                    "type": "state",
                    "agents_total": n_ag,
                    "agents_seen": ag_seen,
                    "n_agents_seen": np.sum(ag_seen),
                    "val": Xnext,
                    "norms_trav": norms_trav,
                }
            )
        df = pd.DataFrame(data)

    df.set_index("timestamp", inplace=True)

    return df


def position_init(
    X, tol_s=0.1
):  # initial agent labeling, assuming at least n_ag points found

    Xnext = np.ones((0, 3))
    for i, xi in enumerate(X):

        if i > 0:
            ds = norm(Xnext - xi, axis=1)
            if np.min(ds) < tol_s:  # spatial filter for too close points
                continue
            j = -1
        else:
            j = 0

        Xnext = np.insert(Xnext, j, xi, axis=0)

    return Xnext


def match_position(X, Xprev, tol_t=0.3):  # matching to previous points

    if X.size == 0:  # if totally empty, return previous
        return (
            Xprev,
            np.nan * np.ones(Xprev.shape[0]),
            np.zeros(Xprev.shape[0], dtype=int),
        )

    Xnext = np.copy(Xprev)  # if not overwritten will preserve old value
    min_norms = np.zeros(Xprev.shape[0])
    ag_seen = np.zeros(Xprev.shape[0], dtype=int)

    for i, xi in enumerate(Xprev):

        ds = norm(X - xi, axis=1)
        # print(ds)
        dix = np.argmin(ds[~np.isnan(ds)])
        min_d = np.min(ds)
        min_norms[i] = ds[dix]

        if ds[dix] > tol_t:  # temporal filter for too far points
            # xj = np.nan*xi # fill w nans?
            continue  # leave old
        else:
            ag_seen[i] = 1
            Xnext[i] = X[dix]

    return Xnext, min_norms, ag_seen


# Plotting function
def plot_data(df, norm_coloring=False, point_cloud=True):
    fig = plt.figure(figsize=(10, 6))
    cmaps = ["Blues", "Oranges", "Greens", "Reds", "Purples"]
    ax = fig.add_subplot(1, 1, 1)

    if point_cloud:
        names = ["s" + str(i + 1) for i in range(df.iloc[0]["agents_total"])]

        vals = np.dstack(df.val.values)
        norms = np.vstack(df.norms_trav).transpose()

        for i in range(len(names)):

            ax.scatter(
                vals[i, 0, 0],
                vals[i, 1, 0],
                s=500,
                marker="*",
                color=mpl.cm.tab10(i),
                label="s" + str(i) + " start",
            )  # , "*", color=mpl.cm.tab10(i), markersize=5., alpha=0.3)

            if norm_coloring:
                ax.plot(
                    vals[i, 0, :],
                    vals[i, 1, :],
                    "-",
                    color=mpl.cm.tab10(i),
                    label="s" + str(i) + " path",
                )
                sc = ax.scatter(
                    vals[i, 0, :],
                    vals[i, 1, :],
                    c=norms[i],
                    cmap=cmaps[i],
                    label="s" + str(i),
                )
                fig.colorbar(sc, ax=ax)
            else:
                # colors = mpl.colormaps[cmaps[i]](np.linspace(0, 1, vals.shape[2]))
                ax.plot(
                    vals[i, 0, :],
                    vals[i, 1, :],
                    "-",
                    color=mpl.cm.tab10(i),
                    label="s" + str(i) + " path",
                )
                sc = ax.scatter(
                    vals[i, 0, :],
                    vals[i, 1, :],
                    color=mpl.cm.tab10(i),
                    alpha=0.2,
                    label="s" + str(i),
                )
                # fig.colorbar(sc, ax=ax) # doesn't work...

    else:
        names = df.names.unique()

        for i in range(len(names)):
            df_state = df.loc[df.name == names[i]]
            # cmapi = mpl.cm.get_cmap(cmaps[i], len(df_state))

            if norm_coloring:
                sc = ax.scatter(
                    [r[0] for r in df_state.val],
                    [r[1] for r in df_state.val],
                    c=df_state.norm_traveled,
                    cmap="jet",
                    label="s" + str(i),
                )
                fig.colorbar(sc, ax=ax)
            else:
                colors = mpl.colormaps[cmaps[i]](np.linspace(0, 1, len(df_state)))
                sc = ax.scatter(
                    [r[0] for r in df_state.val],
                    [r[1] for r in df_state.val],
                    color=colors,
                    label="s" + str(i),
                )

    # WAS 2/13 TODO - colors in plot look like sequence isnt complete? actually maybe it just sits at the end for a bit
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # ax = fig.add_subplot(1,2,1, projection='3d')
    # df_state = df.loc[df['type'] == 'state']
    # colors = mpl.cm.viridis(np.linspace(0, 1, len(df_state)))
    # ax.scatter([r[0] for r in df_state.val], [r[1] for r in df_state.val], [r[2] for r in df_state.val], color=colors)
    # ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    # ax2 = fig.add_subplot(1,2,2)
    # df_control = df.loc[df['type'] == 'control']
    # for i, name in enumerate(['roll', 'pitch', 'yaw_dot', 'thrust']):
    #     ax2.plot(df_control.index, [r[i] for r in df_control.val], label=name)
    # ax2.set_xlabel('Time')

    plt.title("States")
    # plt.legend(names)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot states and controls from a ROS bag."
    )
    parser.add_argument(
        "-f",
        type=str,
        help="Path to the ROS bag file",
        default="two_agent_sphero_test_2024-02-13-17-04-10.bag",
    )
    parser.add_argument("-pc", type=bool, help="Point Cloud Flag", default=True)
    parser.add_argument("-nc", type=bool, help="Norm Coloring Flag", default=False)
    parser.add_argument("-t", type=float, help="Time Filter Parameter", default=0.35)

    args = parser.parse_args()

    df = ros_messages_to_dataframe(args.f, point_cloud=args.pc, tol_t=args.t)
    # df = ros_messages_to_dataframe('2024-02-09-20-21-30.bag', point_cloud=False)
    # df = ros_messages_to_dataframe('two_agent_sphero_test_2024-02-13-17-04-10.bag', point_cloud=True)
    # df = ros_messages_to_dataframe('two_agent_sphero_test_slow2_2024-02-13-17-11-09.bag', point_cloud=True)

    plot_data(df, norm_coloring=args.nc)
