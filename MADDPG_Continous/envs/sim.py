import json
import time
import numpy as np
from envs.timetable import Timetable
from envs.bus import Bus
from envs.route import Route
from envs.station import Station
from envs.visualize import visualize
import pandas as pd
from gym.spaces.box import Box
from gym.spaces import MultiDiscrete
import copy
import os, sys
import pygame
import json
# TODO 归一化
# TODO 对站点标签进行one-hot处理
# TODO 是否会因为最后一个agent经常不发生作用而不被训练的充分，然后出问题？参考下秦伟志的
# TODO 对站点和direction进行one-hot
# TODO 增加一个提前结束的条件
# 已经把relu换成relu6
# 已经归一化state，


class env_bus(object):
    
    def __init__(self, path, debug=False, render=False):
        if render:
            pygame.init()

        self.path = path
        sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
        config_path = os.path.join(path, 'config.json')
        with open(config_path, 'r') as f:
            args = json.load(f)
        self.args = args
        self.effective_trip_num = 264
        
        self.time_step = args["time_step"]
        self.passenger_update_freq = args["passenger_state_update_freq"]
        # read data, multi-index used here
        self.od = pd.read_excel(os.path.join(path, "data/passenger_OD.xlsx"), index_col=[1, 0])
        self.station_set = pd.read_excel(os.path.join(path, "data/stop_news.xlsx"))
        self.routes_set = pd.read_excel(os.path.join(path, "data/route_news.xlsx"))
        self.timetable_set = pd.read_excel(os.path.join(path, "data/time_table.xlsx"))
        # Truncate the original timetable by first 50 trips to reduce the calculation pressure
        self.timetable_set = self.timetable_set.sort_values(by=['launch_time', 'direction'])[:self.effective_trip_num].reset_index(drop=True)
        # add index for timetable
        self.timetable_set['launch_turn'] = range(self.timetable_set.shape[0])
        self.max_agent_num = 25

        self.visualizer = visualize(self, render)

        # Set effective station and time period
        self.effective_station_name = sorted(set([self.od.index[i][0] for i in range(self.od.shape[0])]))
        self.effective_period = sorted(list(set([self.od.index[i][1] for i in range(self.od.shape[0])])))


        if debug:
            self.summary_data = pd.DataFrame(columns=['bus_id', 'station_id', 'trip_id', 'abs_dis', 'forward_headway',
                                                  'backward_headway', 'headway_diff', 'time'])
            self.summary_reward = pd.DataFrame(columns=['bus_id', 'station_id', 'trip_id', 'forward_headway',
                                                    'backward_headway', 'reward', 'time'])

        self.stations = self.set_stations()
        self.routes = self.set_routes()
        self.timetables = self.set_timetables()

        self.state_dim = 3

        self.action_space = Box(0, 20, shape=(1,))
        self.observation_space = Box(0, 60, shape=(self.state_dim,))

    # def save_snapshot(self):
    #     snapshot = {
    #         'current_time': self.current_time,
    #         'bus_all': copy.deepcopy(self.bus_all),
    #         'stations': copy.deepcopy(self.stations),
    #         'routes': copy.deepcopy(self.routes),
    #         'timetables': copy.deepcopy(self.timetables),
    #         'state': copy.deepcopy(self.state),
    #         'reward': copy.deepcopy(self.reward),
    #         'done': self.done
    #     }
    #     return snapshot
    #
    # def load_snapshot(self, snapshot):
    #     self.current_time = snapshot['current_time']
    #     self.bus_all = copy.deepcopy(snapshot['bus_all'])
    #     self.stations = copy.deepcopy(snapshot['stations'])
    #     self.routes = copy.deepcopy(snapshot['routes'])
    #     self.timetables = copy.deepcopy(snapshot['timetables'])
    #     self.state = copy.deepcopy(snapshot['state'])
    #     self.reward = copy.deepcopy(snapshot['reward'])
    #     self.done = snapshot['done']

    # Example usage:
    # env = env_bus(config, os.getcwd(), debug=debug)
    # snapshot = env.save_snapshot()
    # # ... run some steps ...
    # env.load_snapshot(snapshot)

    # def random_actions(self):
    #     return np.random.randint(1, 1 + self.action_dim, self.max_agent_num)

    # return the bus which is in terminal for now (which is not on route)
    @property
    def bus_in_terminal(self):
        return [bus for bus in self.bus_all if not bus.on_route]

    # @property
    # def bus_on_route(self):
    #     return [bus for bus in self.bus_all if bus.on_route]

    def set_timetables(self):
        return [Timetable(self.timetable_set['launch_time'][i], self.timetable_set['launch_turn'][i], self.timetable_set['direction'][i]) for i in range(self.timetable_set.shape[0])]

    def set_routes(self):
        return [
            Route(self.routes_set['route_id'][i], self.routes_set['start_stop'][i], self.routes_set['end_stop'][i],
                  self.routes_set['distance'][i], self.routes_set['V_max'][i], self.routes_set.iloc[i, 5:]) for i in
            range(self.routes_set.shape[0])]

    def set_stations(self):
        station_concat = pd.concat([self.station_set, self.station_set[::-1][1:]]).reset_index()
        total_station = []
        for idx, station in station_concat.iterrows():
            # station type is 0 if Terminal else 1
            station_type = 1 if station['stop_name'] not in ['Terminal_up', 'Terminal_down'] else 0

            direction = False if idx >= station_concat.shape[0] / 2 else True
            od = None
            if station['stop_name'] in self.effective_station_name:
                od = self.od.loc[station['stop_name'], station['stop_name']:] if direction else self.od.loc[station['stop_name'], :station['stop_name']]
                # To reduce the OD value in False direction stations in ['X13','X14','X15'] because too many passengers stuck cause the overwhelming
                if station['stop_name'] in ['X13','X14','X15'] and not direction:
                    od *= 0.4

                od.index = od.index.map(str)
                od = od.to_dict(orient='index')

            total_station.append(Station(station_type, station['stop_id'], station['stop_name'], direction, od))

        return total_station

    # return default state and reward
    def reset(self):

        self.current_time = 0

        # initialize station, routes and timetables
        self.stations = self.set_stations()
        self.routes = self.set_routes()
        self.timetables = self.set_timetables()

        # initial list of bus on route
        self.bus_id = 0
        self.bus_all = []
        self.route_state = []

        # self.state is combine with route_state, which contains the route.speed_limit of each route, station_state, which
        # contains the station.waiting_passengers of each station and bus_state, which is bus.obs for each bus.
        self.state = {key: [] for key in range(self.max_agent_num)}
        self.reward = {key: 0 for key in range(self.max_agent_num)}
        self.done = False

        self.action_dict = {key: None for key in list(range(self.max_agent_num))}

    def initialize_state(self, render=False):
        def count_non_empty_sublist(lst):
            return sum(1 for sublist in lst if sublist)

        while count_non_empty_sublist(list(self.state.values())) == 0:
            self.state, self.reward, _ = self.step(self.action_dict, render=render)

        return self.state, self.reward, self.done

    def launch_bus(self, trip):
        # Trip set(self.timetable) contain both direction trips. So we have to make sure the direction and launch time
        # is satisfied before the trip launched.
        # If there is no more appropriate bus in terminal, create a new bus, then add it to all_bus list.
        if len(list(filter(lambda i: i.direction == trip.direction, self.bus_in_terminal))) == 0:
            # cause bus.next_station， current_route and effective station & routes is defined by @property, so no initialize here
            bus = Bus(self.bus_id, trip.launch_turn, trip.launch_time, trip.direction, self.routes, self.stations)
            self.bus_all.append(bus)
            self.bus_id += 1
        else:
            # if there is bus in terminal and also the direction is satisfied, then we reuse the bus to relaunch one of
            # them, which has the earliest arrived time to terminal.
            bus = sorted(list(filter(lambda i: i.direction == trip.direction, self.bus_in_terminal)), key=lambda bus: bus.back_to_terminal_time)[0]
            bus.reset_bus(trip.launch_turn, trip.launch_time)
            # in drive() function, we set bus.on_route = False when it finished a trip. Here we set it to True because
            # the iteration in drive(), we just update the state of those bus which on routes
            bus.on_route = True

    def step(self, action, debug=False, render=False):
        # Enumerate trips in timetables, if current_time<=launch_time of the trip, then launch it.
        # E.X. timetables = [6:00/launched, 6:05, 6:10], current time is 6:05, then iteration will judge from first trip [6:00]
        # But [6:00] is launched, so next is [6:05]
        for i, trip in enumerate(self.timetables):
            if trip.launch_time <= self.current_time and not trip.launched:
                trip.launched = True
                self.launch_bus(trip)
        # route
        route_state = []
        # update route speed limit by freq
        if self.current_time % self.args['route_state_update_freq'] == 0:
            for route in self.routes:
                route.route_update(self.current_time, self.effective_period)
                route_state.append(route.speed_limit)
            self.route_state = route_state
        # update waiting passengers of every station every second
        # station_state = []
        if self.current_time % self.passenger_update_freq == 0:
            for station in self.stations:
                station.station_update(self.current_time, self.stations, self.passenger_update_freq)
            # station_state.append(len(station.waiting_passengers))
        # update bus state
        for bus in self.bus_all:
            # if bus.bus_id == 0:
                # print(bus.last_station.station_name, bus.absolute_distance)
            # 每次开始前，清零状态和奖励
            bus.reward = None
            bus.obs = []
            if bus.in_station:
                bus.trajectory.append([bus.last_station.station_name, self.current_time, bus.absolute_distance, bus.direction, bus.trip_id])
                bus.trajectory_dict[bus.last_station.station_name].append([bus.last_station.station_name, self.current_time + bus.holding_time, bus.absolute_distance, bus.direction, bus.trip_id])
            if bus.on_route:
                bus.drive(self.current_time, action[bus.bus_id], self.bus_all, debug=debug)

        self.state_bus_list = state_bus_list = list(filter(lambda x: len(x.obs) != 0, self.bus_all))
        self.reward_list = reward_list = list(filter(lambda x: x.reward is not None, self.bus_all))

        if len(state_bus_list) != 0:
            # state_bus_list = sorted(state_bus_list, key=lambda x: x.bus_id)
            for i in range(len(state_bus_list)):
                # print('return state is ', state_bus_list[i].obs, ' for bus: ', state_bus_list[i].bus_id, 'at time:', self.current_time)
                # if len(self.state[state_bus_list[i].bus_id]) < 2:
                self.state[state_bus_list[i].bus_id].append(state_bus_list[i].obs)
                # if state_bus_list[i].last_station.station_id not in [0,1,21,22]:
                #     print(1)
                # else:
                #     self.state[state_bus_list[i].bus_id][0] = self.state[state_bus_list[i].bus_id][1]
                #     self.state[state_bus_list[i].bus_id][1] = state_bus_list[i].obs
                # if state_bus_list[i].bus_id == 0:
                #     print(state_bus_list[i].obs[-1], 'bus_id: ', state_bus_list[i].obs[0], ', station_id: ', state_bus_list[i].obs[1], ', trip_id: ', state_bus_list[i].obs[2])
                #     print('return state is ', state_bus_list[i].obs, ' for bus: ', state_bus_list[i].bus_id,
                #           'at time: ', self.current_time)
                # if len(self.state[state_bus_list[i].bus_id]) > 2:
                #     print(1)
                # if debug:
                #     new_data = [state_bus_list[i].obs[0], state_bus_list[i].obs[1], state_bus_list[i].obs[2],
                #                 state_bus_list[i].obs[4]*1000, state_bus_list[i].obs[6] * 60, state_bus_list[i].obs[7]*60,
                #                 state_bus_list[i].obs[6] * 60 - state_bus_list[i].obs[7] * 60, self.current_time]
                #     self.summary_data.loc[len(self.summary_data)] = new_data
        if len(reward_list) != 0:
            # reward_list = sorted(reward_list, key=lambda x: x.bus_id)
            for i in range(len(reward_list)):
                # if reward_list[i].bus_id == 0:
                #     print('return reward is: ', reward_list[i].reward, ' for bus: ', reward_list[i].bus_id, ' at time:', self.current_time)
                # if (reward_list[i].last_station.station_id != 22 and reward_list[i].direction != 0) and \
                #         (reward_list[i].last_station.station_id != 1 and reward_list[i].direction != 1):
                # if len(self.reward[reward_list[i].bus_id]) > 1:
                #     print(2)
                self.reward[reward_list[i].bus_id] = reward_list[i].reward
                # if debugging:
                #     new_reward = [reward_list[i].bus_id, reward_list[i].last_station.station_id,
                #                   reward_list[i].trip_id, reward_list[i].forward_headway,
                #                   reward_list[i].backward_headway, reward_list[i].reward,
                #                   self.current_time + reward_list[i].holding_time]
                #     self.summary_reward.loc[len(self.summary_reward)] = new_reward

        self.current_time += self.time_step
        unhealthy_all = [bus.is_unhealthy for bus in self.bus_all]
        if sum([trip.launched for trip in self.timetables]) == len(self.timetables) and sum([bus.on_route for bus in self.bus_all]) == 0:
            self.done = True
            for bus in self.bus_all:
                bus.trajectory.clear()  # 清空轨迹列表
                bus.trajectory_dict.clear()  # 清空轨迹字典
                del bus.trajectory  # 强制删除对象，帮助 GC
                del bus.trajectory_dict
            for station in self.stations:
                station.waiting_passengers = np.array([])
                station.total_passenger.clear()
        else:
            self.done = False

        if self.done and debug:
            self.summary_data = self.summary_data.sort_values(['bus_id', 'time'])

            output_dir = os.path.join(self.path, 'pic')
            os.makedirs(output_dir, exist_ok=True)
            self.visualizer.plot()

            self.summary_data.to_csv(os.path.join(output_dir, 'summary_data.csv'))
            self.summary_reward = self.summary_reward.sort_values(['bus_id', 'time'])
            self.summary_reward.to_csv(os.path.join(self.path, 'pic', 'summary_reward.csv'))

        if render and self.current_time % 1 == 0:
            self.visualizer.render()
            time.sleep(0)  # Add a delay to slow down the rendering

        return self.state, self.reward, self.done


if __name__ == '__main__':
    debug = True
    render = True
    if render:
        pygame.init()
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')

    env = env_bus(os.getcwd(), debug=debug)
    start_time = time.time()
    actions = {key: 15. for key in list(range(env.max_agent_num))}
    env.reset()
    while not env.done:
        state, reward, done = env.step(action=actions, debug=debug, render=render)
        # if debug and env.current_time % 4 == 0:
        #     if render:
        #         env.visualizer.render()
        #         time.sleep(0.05)  # Add a delay to slow down the rendering
    pygame.quit()
    print(time.time() - start_time)
