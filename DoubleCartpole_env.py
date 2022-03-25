"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from scipy.integrate import ode

g = 9.8  # gravity
force_mag = 10.0
tau = 0.02  # seconds between state updates

# cart
m_cart = 1

# pole 1
l_1 = 1 # length
m_1 = 0.1  # mass

# pole 2
l_2 = 1  # length
m_2 = 0.1  # mass


def f(time, state, input):
    x = state[0]
    x_dot = state[1]
    theta_1 = state[2]
    theta_1_dot = state[3]
    theta_2 = state[4]
    theta_2_dot = state[5]

    x_dot_dot = ((l_1 * l_2 * m_2 * np.sin(theta_1 - theta_2) * theta_1_dot ** 2
                + g * l_2 * m_2 * np.sin(theta_2)) * (m_1 * np.cos(theta_2) + m_2 * np.cos(theta_2)
                - m_1 * np.cos(theta_1 - theta_2) * np.cos(theta_1)
                - m_2 * np.cos(theta_1 - theta_2) * np.cos(theta_1))) / (l_2 * m_2 ** 2 * np.cos(theta_1 - theta_2) ** 2
                - l_2 * m_2 ** 2 - l_2 * m_1 ** 2 - 2 * l_2 * m_1 * m_2 - l_2 * m_1 * m_cart - l_2 * m_2 * m_cart
                + l_2 * m_1 ** 2 * np.cos(theta_1) ** 2 + l_2 * m_2 ** 2 * np.cos(theta_1) ** 2 + l_2 * m_2 ** 2 * np.cos(theta_2) ** 2
                + l_2 * m_1 * m_2 * np.cos(theta_1 - theta_2) ** 2 + l_2 * m_2 * m_cart * np.cos(theta_1 - theta_2) ** 2
                + 2 * l_2 * m_1 * m_2 * np.cos(theta_1) ** 2 + l_2 * m_1 * m_2 * np.cos(theta_2) ** 2
                - 2 * l_2 * m_2 ** 2 * np.cos(theta_1 - theta_2) * np.cos(theta_1) * np.cos(theta_2)
                - 2 * l_2 * m_1 * m_2 * np.cos(theta_1 - theta_2) * np.cos(theta_1) * np.cos(theta_2)) \
                + ((- l_1 * l_2 * m_2 * np.sin(theta_1 - theta_2) * theta_2_dot ** 2
                + g * l_1 * np.sin(theta_1) * (m_1 + m_2)) * (m_1 * np.cos(theta_1) + m_2 * np.cos(theta_1)
                - m_2 * np.cos(theta_1 - theta_2) * np.cos(theta_2))) / (l_1 * m_2 ** 2 * np.cos(theta_1 - theta_2) ** 2
                - l_1 * m_2 ** 2 - l_1 * m_1 ** 2 - 2 * l_1 * m_1 * m_2 - l_1 * m_1 * m_cart - l_1 * m_2 * m_cart
                + l_1 * m_1 ** 2 * np.cos(theta_1) ** 2 + l_1 * m_2 ** 2 * np.cos(theta_1) ** 2 + l_1 * m_2 ** 2 * np.cos(theta_2) ** 2
                + l_1 * m_1 * m_2 * np.cos(theta_1 - theta_2) ** 2 + l_1 * m_2 * m_cart * np.cos(theta_1 - theta_2) ** 2
                + 2 * l_1 * m_1 * m_2 * np.cos(theta_1) ** 2 + l_1 * m_1 * m_2 * np.cos(theta_2) ** 2
                - 2 * l_1 * m_2 ** 2 * np.cos(theta_1 - theta_2) * np.cos(theta_1) * np.cos(theta_2)
                - 2 * l_1 * m_1 * m_2 * np.cos(theta_1 - theta_2) * np.cos(theta_1) * np.cos(theta_2)) \
                - ((- m_2 * np.cos(theta_1 - theta_2) ** 2 + m_1 + m_2) *(l_1 * np.sin(theta_1) * (m_1 + m_2) * theta_1_dot ** 2
                + l_2 * m_2 * np.sin(theta_2) * theta_2_dot ** 2 + input)) / (m_1 ** 2 * np.cos(theta_1) ** 2 - m_1 * m_cart
                - m_2 * m_cart - 2 * m_1 * m_2 + m_2 ** 2 * np.cos(theta_1) ** 2 + m_2 ** 2 * np.cos(theta_2) ** 2
                + m_2 ** 2 * np.cos(theta_1 - theta_2) ** 2 - m_1 ** 2 - m_2 ** 2 + 2 * m_1 * m_2 * np.cos(theta_1) ** 2
                + m_1 * m_2 * np.cos(theta_2) ** 2 + m_1 * m_2 * np.cos(theta_1 - theta_2) ** 2
                + m_2 * m_cart * np.cos(theta_1 - theta_2) ** 2 - 2 * m_2 ** 2 * np.cos(theta_1 - theta_2) * np.cos(theta_1) * np.cos(theta_2)
                - 2 * m_1 * m_2 * np.cos(theta_1 - theta_2) * np.cos(theta_1) * np.cos(theta_2))

    theta_1_dot_dot = ((m_1 * np.cos(theta_1) + m_2 * np.cos(theta_1)
                - m_2 * np.cos(theta_1 - theta_2) * np.cos(theta_2)) * (l_1 * np.sin(theta_1) * (m_1 + m_2) * theta_1_dot ** 2
                + l_2 * m_2 * np.sin(theta_2) * theta_2_dot ** 2 + input)) \
                / (l_1 * m_2 ** 2 * np.cos(theta_1 - theta_2) ** 2 - l_1 * m_2 ** 2 - l_1 * m_1 ** 2 - 2 * l_1 * m_1 * m_2
                - l_1 * m_1 * m_cart - l_1 * m_2 * m_cart + l_1 * m_1 ** 2 * np.cos(theta_1) ** 2
                + l_1 * m_2 ** 2 * np.cos(theta_1) ** 2 + l_1 * m_2 ** 2 * np.cos(theta_2) ** 2
                + l_1 * m_1 * m_2 * np.cos(theta_1 - theta_2) ** 2 + l_1 * m_2 * m_cart * np.cos(theta_1 - theta_2) ** 2
                + 2 * l_1 * m_1 * m_2 * np.cos(theta_1) ** 2 + l_1 * m_1 * m_2 * np.cos(theta_2) ** 2
                - 2 * l_1 * m_2 ** 2 * np.cos(theta_1 - theta_2) * np.cos(theta_1) * np.cos(theta_2)
                - 2 * l_1 * m_1 * m_2 * np.cos(theta_1 - theta_2) * np.cos(theta_1) * np.cos(theta_2)) \
                - ((- l_1 * l_2 * m_2 * np.sin(theta_1 - theta_2) * theta_2_dot ** 2
                + g * l_1 * np.sin(theta_1) * (m_1 + m_2)) * (- m_2 * np.cos(theta_2) ** 2 + m_1 + m_2 + m_cart)) \
                / (l_1 ** 2 * m_1 ** 2 * np.cos(theta_1) ** 2 - l_1 ** 2 * m_2 ** 2
                - 2 * l_1 ** 2 * m_1 * m_2 - l_1 ** 2 * m_1 * m_cart - l_1 ** 2 * m_2 * m_cart - l_1 ** 2 * m_1 ** 2
                + l_1 ** 2 * m_2 ** 2 * np.cos(theta_1) ** 2 + l_1 ** 2 * m_2 ** 2 * np.cos(theta_2) ** 2
                + l_1 ** 2 * m_2 ** 2 * np.cos(theta_1 - theta_2) ** 2 + 2 * l_1 ** 2 * m_1 * m_2 * np.cos(theta_1) ** 2
                + l_1 ** 2 * m_1 * m_2 * np.cos(theta_2) ** 2 + l_1 ** 2 * m_1 * m_2 * np.cos(theta_1 - theta_2) ** 2
                + l_1 ** 2 * m_2 * m_cart * np.cos(theta_1 - theta_2) ** 2
                - 2 * l_1 ** 2 * m_2 ** 2 * np.cos(theta_1 - theta_2) * np.cos(theta_1) * np.cos(theta_2)
                - 2 * l_1 ** 2 * m_1 * m_2 * np.cos(theta_1 - theta_2) * np.cos(theta_1) * np.cos(theta_2)) \
                + ((l_1 * l_2 * m_2 * np.sin(theta_1 - theta_2) * theta_1_dot ** 2
                + g * l_2 * m_2 * np.sin(theta_2)) * (m_1 * np.cos(theta_1 - theta_2) + m_2 * np.cos(theta_1 - theta_2)
                + m_cart * np.cos(theta_1 - theta_2) - m_1 * np.cos(theta_1) * np.cos(theta_2)
                - m_2 * np.cos(theta_1) * np.cos(theta_2))) / (l_1 * l_2 * m_1 ** 2 * np.cos(theta_1) ** 2 - l_1 * l_2 * m_2 ** 2
                - 2 * l_1 * l_2 * m_1 * m_2 - l_1 * l_2 * m_1 * m_cart - l_1 * l_2 * m_2 * m_cart - l_1 * l_2 * m_1 ** 2
                + l_1 * l_2 * m_2 ** 2 * np.cos(theta_1) ** 2 + l_1 * l_2 * m_2 ** 2 * np.cos(theta_2) ** 2
                + l_1 * l_2 * m_2 ** 2 * np.cos(theta_1 - theta_2) ** 2 + 2 * l_1 * l_2 * m_1 * m_2 * np.cos(theta_1) ** 2
                + l_1 * l_2 * m_1 * m_2 * np.cos(theta_2) ** 2 + l_1 * l_2 * m_1 * m_2 * np.cos(theta_1 - theta_2) ** 2
                + l_1 * l_2 * m_2 * m_cart * np.cos(theta_1 - theta_2) ** 2
                - 2 * l_1 * l_2 * m_2 ** 2 * np.cos(theta_1 - theta_2) * np.cos(theta_1) * np.cos(theta_2)
                - 2 * l_1 * l_2 * m_1 * m_2 * np.cos(theta_1 - theta_2) * np.cos(theta_1) * np.cos(theta_2))

    theta_2_dot_dot = ((- l_1 * l_2 * m_2 * np.sin(theta_1 - theta_2) * theta_2_dot ** 2
                + g * l_1 * np.sin(theta_1) * (m_1 + m_2)) * (m_1 * np.cos(theta_1 - theta_2)
                + m_2 * np.cos(theta_1 - theta_2) + m_cart * np.cos(theta_1 - theta_2) - m_1 * np.cos(theta_1) * np.cos(theta_2)
                - m_2 * np.cos(theta_1) * np.cos(theta_2))) / (l_1 * l_2 * m_1 ** 2 * np.cos(theta_1) ** 2
                - l_1 * l_2 * m_2 ** 2 - 2 * l_1 * l_2 * m_1 * m_2 - l_1 * l_2 * m_1 * m_cart
                - l_1 * l_2 * m_2 * m_cart - l_1 * l_2 * m_1 ** 2 + l_1 * l_2 * m_2 ** 2 * np.cos(theta_1) ** 2
                + l_1 * l_2 * m_2 ** 2 * np.cos(theta_2) ** 2 + l_1 * l_2 * m_2 ** 2 * np.cos(theta_1 - theta_2) ** 2
                + 2 * l_1 * l_2 * m_1 * m_2 * np.cos(theta_1) ** 2 + l_1 * l_2 * m_1 * m_2 * np.cos(theta_2) ** 2
                + l_1 * l_2 * m_1 * m_2 * np.cos(theta_1 - theta_2) ** 2 + l_1 * l_2 * m_2 * m_cart * np.cos(theta_1 - theta_2) ** 2
                - 2 * l_1 * l_2 * m_2 ** 2 * np.cos(theta_1 - theta_2) * np.cos(theta_1) * np.cos(theta_2)
                - 2 * l_1 * l_2 * m_1 * m_2 * np.cos(theta_1 - theta_2) * np.cos(theta_1) * np.cos(theta_2)) \
                - ((l_1 * l_2 * m_2 * np.sin(theta_1 - theta_2) * theta_1_dot ** 2
                + g * l_2 * m_2 * np.sin(theta_2)) * (2 * m_1 * m_2 + m_1 * m_cart + m_2 * m_cart
                - m_1 ** 2 * np.cos(theta_1) ** 2 - m_2 ** 2 * np.cos(theta_1) ** 2 + m_1 ** 2 + m_2 ** 2
                - 2 * m_1 * m_2 * np.cos(theta_1) ** 2)) / (l_2 ** 2 * m_2 ** 3 * np.cos(theta_1) ** 2
                - l_2 ** 2 * m_2 ** 3 + l_2 ** 2 * m_2 ** 3 * np.cos(theta_2) ** 2
                + l_2 ** 2 * m_2 ** 3 * np.cos(theta_1 - theta_2) ** 2 - 2 * l_2 ** 2 * m_1 * m_2 ** 2
                - l_2 ** 2 * m_1 ** 2 * m_2 - l_2 ** 2 * m_2 ** 2 * m_cart - l_2 ** 2 * m_1 * m_2 * m_cart
                + 2 * l_2 ** 2 * m_1 * m_2 ** 2 * np.cos(theta_1) ** 2 + l_2 ** 2 * m_1 ** 2 * m_2 * np.cos(theta_1) ** 2
                + l_2 ** 2 * m_1 * m_2 ** 2 * np.cos(theta_2) ** 2 + l_2 ** 2 * m_1 * m_2 ** 2 * np.cos(theta_1 - theta_2) ** 2
                + l_2 ** 2 * m_2 ** 2 * m_cart * np.cos(theta_1 - theta_2) ** 2
                - 2 * l_2 ** 2 * m_2 ** 3 * np.cos(theta_1 - theta_2) * np.cos(theta_1) * np.cos(theta_2)
                - 2 * l_2 ** 2 * m_1 * m_2 ** 2 * np.cos(theta_1 - theta_2) * np.cos(theta_1) * np.cos(theta_2)) \
                + ((l_1 * np.sin(theta_1) * (m_1 + m_2) * theta_1_dot ** 2
                + l_2 * m_2 * np.sin(theta_2) * theta_2_dot ** 2 + input) * (m_1 * np.cos(theta_2) + m_2 * np.cos(theta_2)
                - m_1 * np.cos(theta_1 - theta_2) * np.cos(theta_1) - m_2 * np.cos(theta_1 - theta_2) * np.cos(theta_1))) \
                / (l_2 * m_2 ** 2 * np.cos(theta_1 - theta_2) ** 2 - l_2 * m_2 ** 2 - l_2 * m_1 ** 2 - 2 * l_2 * m_1 * m_2
                - l_2 * m_1 * m_cart - l_2 * m_2 * m_cart + l_2 * m_1 ** 2 * np.cos(theta_1) ** 2
                + l_2 * m_2 ** 2 * np.cos(theta_1) ** 2 + l_2 * m_2 ** 2 * np.cos(theta_2) ** 2
                + l_2 * m_1 * m_2 * np.cos(theta_1 - theta_2) ** 2 + l_2 * m_2 * m_cart * np.cos(theta_1 - theta_2) ** 2
                + 2 * l_2 * m_1 * m_2 * np.cos(theta_1) ** 2 + l_2 * m_1 * m_2 * np.cos(theta_2) ** 2
                - 2 * l_2 * m_2 ** 2 * np.cos(theta_1 - theta_2) * np.cos(theta_1) * np.cos(theta_2)
                - 2 * l_2 * m_1 * m_2 * np.cos(theta_1 - theta_2) * np.cos(theta_1) * np.cos(theta_2))
    return [x_dot, x_dot_dot, theta_1_dot, theta_1_dot_dot, theta_2_dot, theta_2_dot_dot]


class DoubleCartPoleEnv(gym.Env):
    """
    Description:
        A underactuated double pole is attached to a cart, which moves along
        a frictionless track. The double pendulum starts upright and the goal is to
        prevent it from falling over by accelerating and breaking the cart expediently.
    Observation:
        Type: Box(6)
        Num     Observation               Min                     Max
        0       Cart Position             -2.4 m                  2.4 m
        1       Cart Velocity             -Inf                    Inf
        2       Pole1 Angle               -pi                     +pi
        3       Pole1 Angular Velocity    -Inf                    Inf
        4       Pole2 Angle               -pi                     +pi
        5       Pole2 Angular Velocity    -Inf                    Inf
    Actions:
        Type: Continuous(1)
        Value range: [-10 N, 10 N]
        Dimension: Force
        Note:
        The amount by that velocity is reduced or increased is not
        fixed; it depends on the poles' angles. This is because
        the center of gravity of the pole determines the amount of energy needed
        to move the cart underneath it
    Reward:
        Reward is always None and the user has to define an expedient reward to solve the problem.
    Starting State:
        The angles are initialized with a small deviation of [-3 deg, +3 deg], all other states are initialized to zero.
    Episode Termination:
        Absolute cart position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.max_initial_angle = 3 * 2 * np.pi / 360 # 3 degrees in rad

        # Angle at which to fail the episode
        self.x_threshold = 2.4

        self.pendulum_ode = ode(f).set_integrator("dopri5")

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array([self.x_threshold,
                         np.finfo(np.float32).max,
                         np.pi,
                         np.finfo(np.float32).max,
                         np.pi,
                         np.finfo(np.float32).max],
                        dtype=np.float32)

        self.action_space = spaces.Box(np.array([-10]), np.array([10]))
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.done = None

        self.steps_in_episode = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        if not self.steps_in_episode < 200:
            self.close()
            raise ValueError("Seems like the time has run out. The environment must be reset.")
        elif self.done:
            self.close()
            raise ValueError("The constraints were violated. The environment must be reset.")

        err_msg = "%r (%s) invalid" % (action, type(action))

        x, x_dot, theta_1, theta_1_dot, theta_2, theta_2_dot = self.state

        force = np.clip(action, -10.0, 10.0)

        self.pendulum_ode.set_f_params(force)
        self.state = self.pendulum_ode.integrate(self.pendulum_ode.t + self.tau)
        self.state[2] = (self.state[2] + np.pi) % (2 * np.pi) - np.pi
        self.state[4] = (self.state[4] + np.pi) % (2 * np.pi) - np.pi

        reward = None
        self.done = self.steps_in_episode == 199 or not (-self.x_threshold <= x <= self.x_threshold)
        self.steps_in_episode += 1

        return np.array(self.state), reward, self.done, {}

    def reset(self):
        self.state = np.zeros(6)
        self.state[2] = np.random.uniform(-self.max_initial_angle, self.max_initial_angle)
        self.state[4] = np.random.uniform(-self.max_initial_angle, self.max_initial_angle)
        self.pendulum_ode.set_initial_value(self.state, 0.0)
        self.steps_in_episode = 0
        self.done = False
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 600

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 300  # TOP OF CART
        polewidth = 2.0
        polelen_1 = scale * l_1
        polelen_2 = scale * l_2
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)

            l, r, t, b = -polewidth / 2, polewidth / 2, polelen_1 - polewidth / 2, -polewidth / 2
            pole_1 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole_1.set_color(.8, .6, .4)
            self.poletrans_1 = rendering.Transform(translation=(0, axleoffset))
            pole_1.add_attr(self.poletrans_1)
            pole_1.add_attr(self.carttrans)
            self.viewer.add_geom(pole_1)

            l, r, t, b = -polewidth / 2, polewidth / 2, polelen_2 - polewidth / 2, -polewidth / 2
            pole_2 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole_2.set_color(.8, .9, .4)
            self.poletrans_2 = rendering.Transform(translation=(0, polelen_1))
            pole_2.add_attr(self.poletrans_2)
            pole_2.add_attr(self.carttrans)
            self.viewer.add_geom(pole_2)

            self.weight1 = rendering.make_circle(polewidth * 5)
            self.weighttrans_1 = rendering.Transform(translation=(0, polelen_1))
            self.weight1.add_attr(self.weighttrans_1)
            self.weight1.add_attr(self.poletrans_1)
            self.weight1.add_attr(self.carttrans)
            self.weight1.set_color(.5, .5, .8)
            self.viewer.add_geom(self.weight1)

            self.weight2 = rendering.make_circle(polewidth * 5)
            self.weighttrans_2 = rendering.Transform(translation=(0, polelen_2))
            self.weight2.add_attr(self.weighttrans_2)
            self.weight2.add_attr(self.poletrans_2)
            self.weight2.add_attr(self.carttrans)
            self.weight2.set_color(.5, .5, .8)
            self.viewer.add_geom(self.weight2)

            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom_1 = pole_1
            self._pole_geom_2 = pole_2

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole_1 = self._pole_geom_1
        pole_2 = self._pole_geom_2
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen_1 - polewidth / 2, -polewidth / 2
        pole_1.v = [(l, b), (l, t), (r, t), (r, b)]
        pole_2.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)

        self.poletrans_1.set_rotation(-x[2])
        self.poletrans_2.set_translation(- polelen_1 * np.sin(-x[2]), polelen_1 * np.cos(-x[2]) + cartheight / 4.0)
        self.poletrans_2.set_rotation(-x[4])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None