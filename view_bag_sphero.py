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
def ros_messages_to_dataframe(bag_path, point_cloud=True):
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
    for msg in bag.read_messages(topics=topics):
        # Convert the ROS time to a datetime object
        timestamp = datetime.fromtimestamp(msg.timestamp.to_sec())
        # Extract the data fields you're interested in (e.g., 'state' and 'control')
        # This assumes the message has attributes 'state' and 'control'
        if point_cloud:
            data.append(
                {
                    "timestamp": timestamp,
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
    df.set_index("timestamp", inplace=True)

    if point_cloud:
        data = []

        Xnext = position_init(df.states[0])
        for i in range(1, len(df)):
            Xnext, norms_trav = match_position(df.states[i], Xnext)
            di = {"timestamp": df.index[i], "type": "state"}
            for i, x in dict(enumerate(Xnext)).items():
                di.update({"name": "s" + str(i + 1)})
                di.update({"val": x})
                di.update({"norm_traveled": norms_trav[i]})
            data.append(di)

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
        return Xprev, np.nan * np.ones(Xprev.shape[0])

    Xnext = np.copy(Xprev)  # if not overwritten will preserve old value
    min_norms = np.zeros(Xprev.shape[0])
    for i, xi in enumerate(Xprev):

        ds = norm(X - xi, axis=1)
        print(ds)
        dix = np.argmin(ds[~np.isnan(ds)])
        min_d = np.min(ds)
        min_norms[i] = ds[dix]

        if ds[dix] > tol_t:  # temporal filter for too far points
            # xj = np.nan*xi # fill w nans?
            continue  # leave old
        else:
            Xnext[i] = X[dix]

    return Xnext, min_norms


# Plotting function
def plot_data(df, norm_coloring=True):
    fig = plt.figure(figsize=(10, 6))
    cmaps = ["Blues", "Oranges", "Greens", "Reds", "Yellows"]

    names = df.name.unique()

    ax = fig.add_subplot(1, 1, 1)

    for i in range(len(names)):
        df_state = df.loc[df.name == names[i]]
        # cmapi = mpl.cm.get_cmap(cmaps[i], len(df_state))

        if norm_coloring:
            sc = ax.scatter(
                [r[0] for r in df_state.val],
                [r[1] for r in df_state.val],
                c=df_state.norm_traveled,
                cmap="jet",
            )  # , cmap=cmaps[i])
            fig.colorbar(sc, ax=ax)
        else:
            colors = mpl.colormaps[cmaps[i]](np.linspace(0, 1, len(df_state)))
            ax.scatter(
                [r[0] for r in df_state.val], [r[1] for r in df_state.val], color=colors
            )

    # WAS 2/13 TODO - colors in plot look like sequence isnt complete? actually maybe it just sits at the end for a bit
    ax.legend(names)
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
    # parser = argparse.ArgumentParser(description='Plot states and controls from a ROS bag.')
    # parser.add_argument('bag_path', type=str, help='Path to the ROS bag file')
    # parser.add_argument('pc', type=bool, help='Point Cloud Flag', default=False)
    # args = parser.parse_args()

    # df = ros_messages_to_dataframe(args.bag_path, point_cloud=args.pc)
    # df = ros_messages_to_dataframe('2024-02-09-20-21-30.bag', point_cloud=False)
    df = ros_messages_to_dataframe(
        "two_agent_sphero_test_2024-02-13-17-04-10.bag", point_cloud=True
    )

    plot_data(df, norm_coloring=False)
