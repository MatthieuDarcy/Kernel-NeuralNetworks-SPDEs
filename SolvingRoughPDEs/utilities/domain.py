import numpy as np
import typing
#np.set_printoptions(precision=20)

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

class Interval(object):
    def __init__(self, x1l, x1r):
        self.x1l = x1l
        self.x1r = x1r

    def random_seed(self, s):
        np.random.seed(s)

    def sampling(self, M, M_Omega):
        int_points = np.random.uniform(self.x1l, self.x1r, M_Omega)
        bdry_points = np.array([self.x1l, self.x1r])
        points = np.concatenate((int_points, bdry_points))
        return points

class Square(object):
    def __init__(self, x1l, x1r, x2l, x2r):
        self.x1l = x1l
        self.x1r = x1r
        self.x2l = x2l
        self.x2r = x2r

    def random_seed(self, s):
        np.random.seed(s)

    def sampling(self, M, M_Omega):
        int_points = np.concatenate((np.random.uniform(self.x1l, self.x1r, (M_Omega, 1)), np.random.uniform(self.x2l, self.x2r, (M_Omega, 1))), axis=1)
        bdry_points = np.zeros((M - M_Omega, 2))
        num_per_bdry = int((M - M_Omega) / 4)

        # bottom face
        bdry_points[0:num_per_bdry, 0] = np.random.uniform(self.x1l, self.x1r, num_per_bdry)
        bdry_points[0:num_per_bdry, 1] = self.x2l
        # right face
        bdry_points[num_per_bdry:2 * num_per_bdry, 0] = self.x1r
        bdry_points[num_per_bdry:2 * num_per_bdry, 1] = np.random.uniform(self.x2l, self.x2r, num_per_bdry)
        # top face
        bdry_points[2 * num_per_bdry:3 * num_per_bdry, 0] = np.random.uniform(self.x1l, self.x1r, num_per_bdry)
        bdry_points[2 * num_per_bdry:3 * num_per_bdry, 1] = self.x2r
        # left face
        bdry_points[3 * num_per_bdry:M - M_Omega, 1] = np.random.uniform(self.x2l, self.x2r, M - M_Omega - 3 * num_per_bdry)
        bdry_points[3 * num_per_bdry:M - M_Omega, 0] = self.x1l

        points = np.concatenate((int_points, bdry_points), axis=0)
        return points

class TimeDependentR2DSquare(object):
    def __init__(self, ti, te, x1l, x1r, x2l, x2r):
        self.ti = ti
        self.te = te
        self.x1l = x1l
        self.x1r = x1r
        self.x2l = x2l
        self.x2r = x2r

    def random_seed(self, s):
        np.random.seed(s)

    def sampling(self, M_Omega, M_I, M_T):
        int_points = np.concatenate((np.random.uniform(self.ti, self.te, (M_Omega, 1)), np.random.uniform(self.x1l, self.x1r, (M_Omega, 1)), np.random.uniform(self.x2l, self.x2r, (M_Omega, 1))), axis=1)
        initial_points = np.concatenate((self.ti + np.zeros((M_I, 1)), np.random.uniform(self.x1l, self.x1r, (M_I, 1)), np.random.uniform(self.x2l, self.x2r, (M_I, 1))), axis=1)
        terminal_points = np.concatenate((self.te + np.zeros((M_T, 1)), np.random.uniform(self.x1l, self.x1r, (M_T, 1)), np.random.uniform(self.x2l, self.x2r, (M_T, 1))), axis=1)
        points = np.concatenate((int_points, initial_points, terminal_points), axis=0)
        return points

    def sampling_at_time_t(self, t, M):
        points = np.concatenate((t + np.zeros((M, 1)), np.random.uniform(self.x1l, self.x1r, (M, 1)), np.random.uniform(self.x2l, self.x2r, (M, 1))), axis=1)
        return points

class TimeDependentR1DSquare(object):
    def __init__(self, ti, te, x1l, x1r):
        self.ti = ti
        self.te = te
        self.x1l = x1l
        self.x1r = x1r

    def random_seed(self, s):
        np.random.seed(s)

    @typing.overload
    def sampling(self, M_Omega, M_I, M_T):
        int_points = np.concatenate((np.random.uniform(self.ti, self.te, (M_Omega, 1)), np.random.uniform(self.x1l, self.x1r, (M_Omega, 1))), axis=1)
        initial_points = np.concatenate((self.ti + np.zeros((M_I, 1)), np.random.uniform(self.x1l, self.x1r, (M_I, 1))), axis=1)
        terminal_points = np.concatenate((self.te + np.zeros((M_T, 1)), np.random.uniform(self.x1l, self.x1r, (M_T, 1))), axis=1)
        points = np.concatenate((int_points, initial_points, terminal_points), axis=0)
        return points

    def sampling(self, M_Omega, M_I):
        int_points = np.concatenate((np.random.uniform(self.ti, self.te, (M_Omega, 1)), np.random.uniform(self.x1l, self.x1r, (M_Omega, 1))), axis=1)
        initial_points = np.concatenate((self.ti + np.zeros((M_I, 1)), np.random.uniform(self.x1l, self.x1r, (M_I, 1))), axis=1)
        points = np.concatenate((int_points, initial_points), axis=0)
        return points

    # def sampling(self, M_Omega, M_I, M_T):
    #     N = 30
    #     h = 4 / N
    #     XX = np.linspace(-2 + h / 2, 2 - h / 2, N)
    #     SEG = 30
    #     dms = XX
    #     dt = 1 / SEG
    #     ts = np.linspace(dt, self.te, SEG)  # jnp.concatenate((jnp.arange(0, SEG) / SEG, jnp.array([1])))
    #     XX2d, YY2d = np.meshgrid(ts, dms)
    #     int_points = np.concatenate((XX2d.reshape(-1, 1), YY2d.reshape(-1, 1)), axis=1)
    #
    #     # int_points = np.concatenate((np.random.uniform(self.ti, self.te, (M_Omega, 1)), np.random.uniform(self.x1l, self.x1r, (M_Omega, 1))), axis=1)
    #     initial_points = np.concatenate((self.ti + np.zeros((M_I, 1)), np.random.uniform(self.x1l, self.x1r, (M_I, 1))), axis=1)
    #     terminal_points = np.concatenate((self.te + np.zeros((M_T, 1)), np.random.uniform(self.x1l, self.x1r, (M_T, 1))), axis=1)
    #     points = np.concatenate((int_points, initial_points, terminal_points), axis=0)
    #
    #     l, r = int_points.shape
    #     return points, l

    def sampling_at_time_t(self, t, M):
        points = np.concatenate((t + np.zeros((M, 1)), np.random.uniform(self.x1l, self.x1r, (M, 1))), axis=1)
        return points


class TimeDependentR1D(object):
    def __init__(self, ti, te, x1l, x1r):
        self.ti = ti
        self.te = te
        self.x1l = x1l
        self.x1r = x1r

    def random_seed(self, s):
        np.random.seed(s)

    def sampling(self, M_time, M_Omega):
        time_points = np.linspace(self.ti, self.te, M_time)
        # points at the initial time
        points_i = np.concatenate((self.ti + np.zeros((M_Omega, 1)), np.random.uniform(self.x1l, self.x1r, (M_Omega, 1))), axis=1)
        # points at the terminal time
        points_e = np.concatenate((self.te + np.zeros((M_Omega, 1)), np.random.uniform(self.x1l, self.x1r, (M_Omega, 1))), axis=1)
        points = np.concatenate((points_i, points_e), axis=0)
        for t in reversed(time_points[1:M_time-1]):
            points_t = np.concatenate((t + np.zeros((M_Omega, 1)), np.random.uniform(self.x1l, self.x1r, (M_Omega, 1))), axis=1)
            points = np.concatenate((points_t, points), axis=0)

        return points

    def sampling_at_time_t(self, t, M):
        points = np.concatenate((t + np.zeros((M, 1)), np.random.uniform(self.x1l, self.x1r, (M, 1))), axis=1)
        return points

class TimeDependentR2D(object):
    def __init__(self, ti, te, x1l, x1r, x2l, x2r):
        self.ti = ti
        self.te = te
        self.x1l = x1l
        self.x1r = x1r
        self.x2l = x2l
        self.x2r = x2r

    def random_seed(self, s):
        np.random.seed(s)

    def sampling(self, M_time, M_Omega):
        time_points = np.linspace(self.ti, self.te, M_time)
        # points at the initial time
        points_i = np.concatenate((self.ti + np.zeros((M_Omega, 1)),
                                   np.random.uniform(self.x1l, self.x1r, (M_Omega, 1)),
                                   np.random.uniform(self.x2l, self.x2r, (M_Omega, 1))), axis=1)
        # points at the terminal time
        points_e = np.concatenate((self.te + np.zeros((M_Omega, 1)),
                                   np.random.uniform(self.x1l, self.x1r, (M_Omega, 1)),
                                   np.random.uniform(self.x2l, self.x2r, (M_Omega, 1))), axis=1)
        points = np.concatenate((points_i, points_e), axis=0)
        for t in reversed(time_points[1:M_time-1]):
            points_t = np.concatenate((t + np.zeros((M_Omega, 1)),
                                         np.random.uniform(self.x1l, self.x1r, (M_Omega, 1)),
                                         np.random.uniform(self.x2l, self.x2r, (M_Omega, 1))), axis=1)
            points = np.concatenate((points_t, points), axis=0)

        return points
        # int_points = np.concatenate((np.random.uniform(self.ti, self.te, (M_Omega, 1)), np.random.uniform(self.x1l, self.x1r, (M_Omega, 1)), np.random.uniform(self.x2l, self.x2r, (M_Omega, 1))), axis=1)
        # initial_points = np.concatenate((self.ti + np.zeros((M_I, 1)), np.random.uniform(self.x1l, self.x1r, (M_I, 1)), np.random.uniform(self.x2l, self.x2r, (M_I, 1))), axis=1)
        # terminal_points = np.concatenate((self.te + np.zeros((M_T, 1)), np.random.uniform(self.x1l, self.x1r, (M_T, 1)), np.random.uniform(self.x2l, self.x2r, (M_T, 1))), axis=1)
        # points = np.concatenate((int_points, initial_points, terminal_points), axis=0)
        #return points

    def sampling_at_time_t(self, t, M):
        points = np.concatenate((t + np.zeros((M, 1)), np.random.uniform(self.x1l, self.x1r, (M, 1)), np.random.uniform(self.x2l, self.x2r, (M, 1))), axis=1)
        return points

class TimeDependentSquare(object):
    def __init__(self, x1l, x1r, x2l, x2r):
        self.x1l = x1l
        self.x1r = x1r
        self.x2l = x2l
        self.x2r = x2r

    def random_seed(self, s):
        np.random.seed(s)

    def sampling(self, M, M_Omega):
        int_points = np.concatenate((np.random.uniform(self.x1l, self.x1r, (M_Omega, 1)), np.random.uniform(self.x2l, self.x2r, (M_Omega, 1))), axis=1)
        bdry_points = np.zeros((M - M_Omega, 2))
        num_per_bdry = int((M - M_Omega) / 3)
        #bdry_points = np.zeros((num_per_bdry * 3, 2))
        #num_per_bdry = M - M_Omega

        # bottom face
        bdry_points[0:num_per_bdry, 0] = self.x1l
        bdry_points[0:num_per_bdry, 1] = np.random.uniform(self.x2l, self.x2r, num_per_bdry)
        # # right face
        bdry_points[num_per_bdry:2 * num_per_bdry, 0] = np.random.uniform(self.x1l, self.x1r, num_per_bdry)
        bdry_points[num_per_bdry:2 * num_per_bdry, 1] = self.x2r
        # left face
        bdry_points[2 * num_per_bdry:, 0] = np.random.uniform(self.x1l, self.x1r, M - M_Omega - 2 * num_per_bdry)
        bdry_points[2 * num_per_bdry:, 1] = self.x2l

        points = np.concatenate((int_points, bdry_points), axis=0)
        return points


class Torus1D(object):
    def __init__(self):
        pass

    def random_seed(self, s):
        np.random.seed(s)

    def sampling(self, M, M_Omega):
        points = np.random.uniform(0, 1, M_Omega)
        # points = np.append(points, 0)
        # points = np.append(points, 1)
        return points

class Torus2D(object):
    def __init__(self):
        pass

    def random_seed(self, s):
        np.random.seed(s)

    def sampling(self, M_Omega):
        points = np.concatenate((np.random.uniform(0, 1, (M_Omega, 1)), np.random.uniform(0, 1, (M_Omega, 1))), axis=1)
        return points

class Torus4D(object):
    def __init__(self):
        pass

    def random_seed(self, s):
        np.random.seed(s)

    def sampling(self, M, M_Omega):
        points = np.concatenate((np.random.uniform(0, 1, (M_Omega, 1)), np.random.uniform(0, 1, (M_Omega, 1)), np.random.uniform(0, 1, (M_Omega, 1)), np.random.uniform(0, 1, (M_Omega, 1))), axis=1)
        return points

# a square with static bars as obstacles in the interior for the congestion model
# the domain has two exits at the bottom
class SquareWithStaticBars(object):
    def __init__(self, T, x0, y0, L, door_l, bar_l, bar_w, bars_num):
        self.T = T #terminal time
        self.x0 = x0 #the x coordinate of the start point
        self.y0 = y0 #the y coordinate of the start point
        self.L = L   #the length of the squre
        self.door_l = door_l #the length of the door
        self.bar_l = bar_l #the lengh of the static bar
        self.bar_w = bar_w #the width of the static bar
        self.bars_num = bars_num #the number of the static bars

        self.bar_start_x = x0 + (L - bar_l) / 2
        self.bar_start_y = y0 + (L - (2 * (bars_num - 1) + 1) * bar_w) / 2

    def sample_space(self, M_int_L, M_int_R, M_int_D, M_int_U, M_int_Mid, M_GammaD, M_GammaN_L, M_GammaN_R, M_GammaN_D, M_GammaN_U):
        left_part_l = (self.L - self.bar_l) / 2
        left_int_points = np.concatenate((np.random.uniform(self.x0, self.x0 + left_part_l, (M_int_L, 1)), np.random.uniform(self.y0, self.y0 + self.L, (M_int_L, 1))), axis=1)
        right_int_points = np.concatenate((np.random.uniform(self.x0 + self.L - left_part_l, self.x0 + self.L, (M_int_R, 1)), np.random.uniform(self.y0, self.y0 + self.L, (M_int_R, 1))), axis=1)
        down_part_w = (self.L - (2 * (self.bars_num - 1) + 1) * self.bar_w) / 2
        down_int_points = np.concatenate((np.random.uniform(self.x0 + left_part_l, self.x0 + left_part_l + self.bar_l, (M_int_D, 1)), np.random.uniform(self.y0, self.y0 + down_part_w, (M_int_D, 1))), axis=1)
        up_int_points = np.concatenate((np.random.uniform(self.x0 + left_part_l, self.x0 + left_part_l + self.bar_l, (M_int_U, 1)), np.random.uniform(self.y0 + self.L - down_part_w, self.y0 + self.L, (M_int_U, 1))), axis=1)

        int_points = np.concatenate((left_int_points, right_int_points, down_int_points, up_int_points), axis=0)
        points_per_bar = int(M_int_Mid / (self.bars_num - 1))
        for i in range(self.bars_num - 1):
            mid_int_points = np.concatenate((np.random.uniform(self.bar_start_x, self.bar_start_x + self.bar_l, (points_per_bar, 1)), np.random.uniform(self.bar_start_y + (2 * i + 1) * self.bar_w, self.bar_start_y + (2 * i + 2) * self.bar_w, (points_per_bar, 1))), axis=1)
            int_points = np.concatenate((int_points, mid_int_points), axis=0)

        return int_points

    def plot_space_domain(self, fig):
        plt.figure(fig.number)
        plt.axes()
        whole_space = plt.Rectangle((self.x0, self.y0), self.L, self.L, fc='pink', zorder=1)

        left_exit = plt.Rectangle((self.x0, self.y0 - 2), self.door_l, 2, fc='red', zorder=1)
        right_exit = plt.Rectangle((self.x0 + self.L - self.door_l, self.y0 - 2), self.door_l, 2, fc='red', zorder=1)
        plt.gca().add_patch(whole_space)
        for i in range(self.bars_num):
            bar = plt.Rectangle((self.bar_start_x, self.bar_start_y + 2 * i * self.bar_w), self.bar_l, self.bar_w, fc='orange', zorder=1)
            plt.gca().add_patch(bar)
        plt.gca().add_patch(left_exit)
        plt.gca().add_patch(right_exit)
        plt.axis('scaled')

    def plot_samples(self, fig, samples):
        plt.figure(fig.number)
        int_data = plt.scatter(samples[:, 0], samples[:, 1], marker="x", label='Interior nodes', zorder=2)
        int_data.set_clip_on(False)
        plt.legend(loc="upper right")
        plt.title('Collocation points')