__author__ = "Yuvaram singh"
__credits__ = ["Yuvaram singh", "rmgi"]
__email__ = "vmvijayayuvaram.s@hcl.com"


import os
import pygame
from math import sin, radians, degrees, copysign
import math
from pygame.math import Vector2
import cv2
import numpy as np
from rl_agent import RL_agent
import copy


class Car:
    def __init__(
        self,
        x,
        y,
        angle=0.0,
        length=4,
        max_steering=30,
        max_acceleration=5.0,
        shadow=None,
    ):
        self.position = Vector2(x, y)
        self.velocity = Vector2(0.0, 0.0)
        self.angle = angle
        self.length = length
        self.max_acceleration = max_acceleration
        self.max_steering = max_steering
        self.max_velocity = 20
        self.brake_deceleration = 20
        self.free_deceleration = 2
        self.acceleration = 0.0
        self.steering = 0.0
        self.lidar = []
        self.lidar_end_pts = []
        self.shadow = shadow
        self.ppu = 32
        self.lidar_ray_length = self.ppu * 5
        self.reset = False

    def reset_state(self):
        self.reset = True
        self.car_pos_center_in_pixel = self.car_pos_center_old_in_pixel
        self.velocity = Vector2(0.0, 0.0)
        self.acceleration = 0.0
        self.steering = 0.0

    def update(self, dt):
        self.velocity += (self.acceleration * dt, 0)
        self.velocity.x = max(
            -self.max_velocity, min(self.velocity.x, self.max_velocity)
        )

        if self.steering:
            turning_radius = self.length / sin(radians(self.steering))
            angular_velocity = self.velocity.x / turning_radius
        else:
            angular_velocity = 0
        self.position += self.velocity.rotate(-self.angle) * dt
        self.angle += degrees(angular_velocity) * dt
        self.car_pos_center_in_pixel = self.position * self.ppu
        length = 30
        end_point_x = length * math.cos(math.radians(180 - self.angle))
        end_point_y = length * math.sin(math.radians(180 - self.angle))
        self.bumper_front = (
            int(self.car_pos_center_in_pixel[0] + end_point_x),
            int(self.car_pos_center_in_pixel[1] + end_point_y),
        )
        end_point_x = length * math.cos(math.radians(360 - self.angle))
        end_point_y = length * math.sin(math.radians(360 - self.angle))
        self.bumper_back = (
            int(self.car_pos_center_in_pixel[0] + end_point_x),
            int(self.car_pos_center_in_pixel[1] + end_point_y),
        )

        if (
            self.shadow[int(self.bumper_front[1]), int(self.bumper_front[0])] == 0
            or self.shadow[int(self.bumper_back[1]), int(self.bumper_back[0])] == 0
        ):

            self.reset = True
            self.car_pos_center_in_pixel = self.car_pos_center_old_in_pixel
            self.velocity = Vector2(0.0, 0.0)
            self.acceleration = 0.0
            self.steering = 0.0
        else:
            self.car_pos_center_old_in_pixel = self.car_pos_center_in_pixel

        angle_list = [
            360 - self.angle - 10,
            360 - self.angle - 20,
            360 - self.angle - 30,
            360 - self.angle - 45,
            360 - self.angle - 90,
            360 - self.angle - 120,
            360 - self.angle - 130,
            360 - self.angle - 145,  # leftside
            360 - self.angle,
            360 - self.angle,
            360 - self.angle + 10,
            360 - self.angle + 20,
            360 - self.angle + 30,
            360 - self.angle + 45,
            360 - self.angle + 90,
            360 - self.angle + 120,
            360 - self.angle + 130,
            360 - self.angle + 145,
        ]  # rightside

        end_point_list = []
        lidar_len = []
        for i in angle_list:
            length = 0
            while True:
                length += 2
                end_point_y = length * math.sin(math.radians(i))
                end_point_x = length * math.cos(math.radians(i))
                end_point = (
                    self.car_pos_center_in_pixel[0] + end_point_x,
                    self.car_pos_center_in_pixel[1] + end_point_y,
                )

                if self.shadow[int(end_point[1]), int(end_point[0])] == 0:

                    end_point_list.append(end_point)
                    lidar_len.append(length / self.ppu)
                    break
                elif length > self.lidar_ray_length:
                    end_point_list.append(end_point)
                    lidar_len.append(self.lidar_ray_length / self.ppu)
                    break
            self.lidar_end_pts = end_point_list

        return (
            lidar_len,
            [self.position[0], self.position[1]],
            self.angle,
            self.acceleration,
            [self.velocity[0], self.velocity[1]],
            self.steering,
        )


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Car tutorial")
        self.width = 1000
        self.height = 750
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.ticks = 30
        self.episode = 0
        self.exit = False
        self.bg = pygame.image.load("./static/track_resized.png")
        self.bg_track = cv2.imread("./static/track_edit.png")
        color_select = np.zeros_like(self.bg_track[:, :, 0])
        img_thresh = (
            (self.bg_track[:, :, 0] > 155)
            & (self.bg_track[:, :, 0] < 165)
            & (self.bg_track[:, :, 1] > 165)
            & (self.bg_track[:, :, 1] < 175)
            & (self.bg_track[:, :, 2] > 160)
            & (self.bg_track[:, :, 2] < 167)
        )
        color_select[img_thresh] = 1
        self.shadow = color_select

        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, "./static/car_resized.png")

        self.car_image = pygame.image.load(image_path)
        self.car = Car(
            2.4,
            13,
            angle=90.0,
            length=1,
            max_steering=30,
            max_acceleration=2.50,
            shadow=self.shadow,
        )
        self.ppu = 32
        self.reset = self.car.reset

        self.Tryout = False
        self.RL = False
        self.velocitystorage = []
        self.oscillation_counter = 0

    def text_objects(self, text, font):
        textSurface = font.render(text, True, (0, 0, 0))
        return textSurface, textSurface.get_rect()

    def button(self, msg, x, y, w, h, ic, ac, action):
        mouse = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()
        if x + w > mouse[0] > x and y + h > mouse[1] > y:
            pygame.draw.rect(self.screen, ac, (x, y, w, h))

            if click[0] == 1:
                if action == "TRYOUT":
                    self.intro = False
                    self.Tryout = True
                    self.RL_pre = False
                elif action == "RL":
                    self.intro = False
                    self.RL = True
                    self.RL_pre = False
                elif action == "RL_PRETRAIN":
                    self.intro = False
                    self.RL = False
                    self.RL_pre = True
        else:
            pygame.draw.rect(self.screen, ic, (x, y, w, h))

        smallText = pygame.font.Font("freesansbold.ttf", 20)
        textSurf, textRect = self.text_objects(msg, smallText)
        textRect.center = ((x + (w / 2)), (y + (h / 2)))
        self.screen.blit(textSurf, textRect)

    def start_screen(self):
        self.intro = True
        while self.intro:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

            self.screen.fill((255, 255, 255))
            largeText = pygame.font.Font("freesansbold.ttf", 115)
            TextSurf, TextRect = self.text_objects("Race Track !", largeText)
            TextRect.center = ((self.width / 2), (self.height / 2))

            self.screen.blit(TextSurf, TextRect)

            self.button("TRYOUT", 250, 450, 100, 50, (0, 200, 0), (0, 255, 0), "TRYOUT")
            self.button("RL", 450, 450, 100, 50, (200, 0, 0), (255, 0, 0), "RL")
            self.button(
                "PRETRAINED", 650, 450, 150, 50, (0, 0, 200), (0, 0, 255), "RL_PRETRAIN"
            )
            pygame.display.flip()
            self.clock.tick(self.ticks)

    def helper(self, observation):
        ldr = []
        for i in observation["Lidar"]:
            ldr.extend([ii for ii in i])

        pos = []
        for i in observation["position"]:
            pos.extend([ii for ii in i])

        ang = []

        ang.extend([ii for ii in observation["angle"]])

        acc = []

        acc.extend([ii for ii in observation["acceleration"]])

        vel = []
        for i in observation["velocity"]:
            vel.extend([ii for ii in i])

        ste = []

        ste.extend([ii for ii in observation["steering"]])

        return {
            "Lidar": ldr,
            "position": pos,
            "angle": ang,
            "acceleration": acc,
            "velocity": vel,
            "steering": ste,
        }

    def check_car_position(self, velocity_x):
        if len(self.velocitystorage) <= 50:
            self.velocitystorage.reverse()
            self.velocitystorage.append(velocity_x)
        else:
            self.velocitystorage.reverse()
            self.velocitystorage.append(velocity_x)
            self.velocitystorage.reverse()
            self.velocitystorage.pop()

            vel_checker = np.array(self.velocitystorage)
            vel_sum = np.sum(vel_checker ** 2)
            print("velocity sum ", vel_sum / 50)
            print("velocity sum ", vel_sum)
            if vel_sum / 50 < 0.02:
                self.oscillation_counter += 1
            else:
                self.oscillation_counter = 0

    def Rl_setup(self, env_requirment):
        assert isinstance(env_requirment, dict), "env_requirment should be a dict"
        assert isinstance(env_requirment["History"], int), "History should be a int"
        assert isinstance(
            env_requirment["replay_buffer"], int
        ), "env_requirment should be a int"
        assert isinstance(
            env_requirment["learning_rate"], float
        ), "env_requirment should be a float"
        assert isinstance(
            env_requirment["max_no_of_episodes"], int
        ), "max_no_of_episodes should be a int"
        assert isinstance(
            env_requirment["update_no_of_episodes"], int
        ), "update_no_of_episodes should be a int"
        self.env_requirment = env_requirment
        self.boot = True

    def Rl(self, agent):
        while True:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit = True
                    pygame.quit()

            if self.reset or self.boot:
                if self.reset:

                    self.velocitystorage = []
                    self.reset = False
                self.episode += 1
                agent.episode = self.episode

                accumulated_reward = 0.0
                if self.boot:
                    self.boot = False

                self.car = Car(
                    2.4,
                    13,
                    angle=90.0,
                    length=1,
                    max_steering=30,
                    max_acceleration=2.50,
                    shadow=self.shadow,
                )
                (
                    lidar_end_pts,
                    position,
                    angle,
                    acceleration,
                    velocity,
                    steering,
                ) = self.update_RL("BREAK")

                self.observation = {
                    "Lidar": [copy.deepcopy(lidar_end_pts)]
                    * self.env_requirment["History"],
                    "position": [copy.deepcopy(position)]
                    * self.env_requirment["History"],
                    "angle": [copy.deepcopy(angle)] * self.env_requirment["History"],
                    "acceleration": [copy.deepcopy(acceleration)]
                    * self.env_requirment["History"],
                    "velocity": [copy.deepcopy(velocity)]
                    * self.env_requirment["History"],
                    "steering": [copy.deepcopy(steering)]
                    * self.env_requirment["History"],
                }
                continue

            obs = self.helper(self.observation)
            action = agent.take_action(observation=obs)
            (
                lidar_end_pts,
                position,
                angle,
                acceleration,
                velocity,
                steering,
            ) = self.update_RL(action)
            if not agent.Pre_trained:

                ld = self.observation["Lidar"]
                ld.reverse()
                ld.append(lidar_end_pts)
                ld.reverse()
                ld.pop()
                self.observation["Lidar"] = ld

                pos = self.observation["position"]
                pos.reverse()
                pos.append(position)
                pos.reverse()
                pos.pop()
                self.observation["position"] = pos

                ang = self.observation["angle"]
                ang.reverse()
                ang.append(angle)
                ang.reverse()
                ang.pop()
                self.observation["angle"] = ang

                acc = self.observation["acceleration"]
                acc.reverse()
                acc.append(acceleration)
                acc.reverse()
                acc.pop()
                self.observation["acceleration"] = acc

                vel = self.observation["velocity"]
                vel.reverse()
                vel.append(velocity)
                vel.reverse()
                vel.pop()
                self.observation["velocity"] = vel

                ste = self.observation["steering"]
                ste.reverse()
                ste.append(steering)
                ste.reverse()
                ste.pop()
                self.observation["steering"] = ste

                print(velocity[0])
                self.check_car_position(velocity[0])

                step_reward = agent.reward(
                    lidar_end_pts,
                    position,
                    angle,
                    acceleration,
                    velocity,
                    steering,
                    crashed=self.reset,
                    previous_obs=obs,
                )
                new_state = agent.observation_parser(
                    obs_dict=self.helper(self.observation)
                )
                agent.step(
                    state=agent.current_state,
                    action=action,
                    reward=step_reward,
                    state_=new_state,
                    terminate=1 if self.reset else 0,
                )
                accumulated_reward += step_reward

                if self.oscillation_counter > 10:
                    self.reset = True
                    self.car.reset_state()
                    self.oscillation_counter = 0
                if self.reset:
                    print(f"Episode {agent.episode} has ended ")
                    print("game reset ", self.reset)
                    if (
                        agent.episode % self.env_requirment["update_no_of_episodes"]
                        == 0
                    ):
                        agent.train()
            else:
                self.check_car_position(velocity[0])
                if self.oscillation_counter > 10:
                    self.reset = True
                    self.car.reset_state()
                    self.oscillation_counter = 0

    def update_RL(self, action):

        ppu = 32
        dt = 56.0 / 1000.0
        if action == "UP":

            if self.car.velocity.x < 0:
                self.car.acceleration = self.car.brake_deceleration
            else:
                self.car.acceleration += 1 * dt
        elif action == "DOWN":
            if self.car.velocity.x > 0:
                self.car.acceleration = -self.car.brake_deceleration
            else:
                self.car.acceleration -= 1 * dt
        elif action == "BREAK":
            if abs(self.car.velocity.x) > dt * self.car.brake_deceleration:
                self.car.acceleration = -copysign(
                    self.car.brake_deceleration, self.car.velocity.x
                )
            else:
                self.car.acceleration = -self.car.velocity.x / dt
        elif action == "RIGHT":
            self.car.acceleration = 0.4

            self.car.steering -= 30 * dt
        elif action == "LEFT":
            self.car.acceleration = 0.4
            self.car.steering += 30 * dt
        else:
            self.car.steering = 0

            if abs(self.car.velocity.x) > dt * self.car.free_deceleration:
                self.car.acceleration = -copysign(
                    self.car.free_deceleration, self.car.velocity.x
                )
            else:
                if dt != 0:
                    self.car.acceleration = -self.car.velocity.x / dt
        self.car.acceleration = max(
            -self.car.max_acceleration,
            min(self.car.acceleration, self.car.max_acceleration),
        )

        self.car.steering = max(
            -self.car.max_steering, min(self.car.steering, self.car.max_steering)
        )
        (
            lidar_end_pts,
            position,
            angle,
            acceleration,
            velocity,
            steering,
        ) = self.car.update(dt)

        self.screen.blit(self.bg, (0, 0))

        rotated = pygame.transform.rotate(self.car_image, self.car.angle)
        rect = rotated.get_rect()
        car_pos_in_pixel = self.car.position * ppu - (rect.width / 2, rect.height / 2)
        car_pos_center_in_pixel = self.car.position * ppu

        self.screen.blit(rotated, car_pos_in_pixel)

        for i in self.car.lidar_end_pts:
            pygame.draw.line(self.screen, (0, 0, 0), car_pos_center_in_pixel, i)
        pygame.draw.line(
            self.screen, (255, 0, 0), car_pos_center_in_pixel, self.car.bumper_back
        )
        pygame.draw.line(
            self.screen, (0, 255, 0), car_pos_center_in_pixel, self.car.bumper_front
        )

        pygame.display.flip()

        self.clock.tick(self.ticks)

        self.reset = self.car.reset
        print("this is car reset ", self.car.reset)
        print("this is game reset ", self.reset)
        return lidar_end_pts, position, angle, acceleration, velocity, steering

    def run(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, "./static/car_resized.png")

        car_image = pygame.image.load(image_path)
        car = Car(
            2.4,
            13,
            angle=90.0,
            length=1,
            max_steering=30,
            max_acceleration=2.50,
            shadow=self.shadow,
        )
        ppu = 32

        while not self.exit:
            pressed = pygame.key.get_pressed()

            reset_check = pressed[pygame.K_r]
            if reset_check:

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.exit = True
                        pygame.quit()

                car = Car(
                    2.4,
                    13,
                    angle=90.0,
                    length=1,
                    max_steering=30,
                    max_acceleration=2.50,
                    shadow=self.shadow,
                )
                dt = self.clock.get_time() / 1000
            else:

                dt = 56.0 / 1000.0

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.exit = True
                        exit()

                if pressed[pygame.K_UP]:

                    if car.velocity.x < 0:
                        car.acceleration = car.brake_deceleration
                    else:
                        car.acceleration += 1 * dt
                elif pressed[pygame.K_DOWN]:

                    if car.velocity.x > 0:
                        car.acceleration = -car.brake_deceleration
                    else:
                        car.acceleration -= 1 * dt
                elif pressed[pygame.K_SPACE]:
                    if abs(car.velocity.x) > dt * car.brake_deceleration:
                        car.acceleration = -copysign(
                            car.brake_deceleration, car.velocity.x
                        )
                    else:
                        car.acceleration = -car.velocity.x / dt
                elif pressed[pygame.K_RIGHT]:
                    car.acceleration = 0.4

                    car.steering -= 30 * dt
                elif pressed[pygame.K_LEFT]:
                    car.acceleration = 0.4
                    car.steering += 30 * dt
                else:
                    car.steering = 0

                    if abs(car.velocity.x) > dt * car.free_deceleration:
                        car.acceleration = -copysign(
                            car.free_deceleration, car.velocity.x
                        )
                    else:
                        if dt != 0:
                            car.acceleration = -car.velocity.x / dt
                car.acceleration = max(
                    -car.max_acceleration, min(car.acceleration, car.max_acceleration)
                )

                car.steering = max(
                    -car.max_steering, min(car.steering, car.max_steering)
                )

            car.update(dt)

            self.screen.blit(self.bg, (0, 0))

            rotated = pygame.transform.rotate(car_image, car.angle)
            rect = rotated.get_rect()

            car_pos_in_pixel = car.position * ppu - (rect.width / 2, rect.height / 2)
            car_pos_center_in_pixel = car.position * ppu

            self.screen.blit(rotated, car_pos_in_pixel)

            for i in car.lidar_end_pts:
                pygame.draw.line(self.screen, (0, 0, 0), car_pos_center_in_pixel, i)
            pygame.draw.line(
                self.screen, (255, 0, 0), car_pos_center_in_pixel, car.bumper_back
            )
            pygame.draw.line(
                self.screen, (0, 255, 0), car_pos_center_in_pixel, car.bumper_front
            )

            pygame.display.flip()

            self.clock.tick(self.ticks)
        pygame.quit()


if __name__ == "__main__":
    game = Game()

    game.start_screen()
    if game.Tryout:
        game.run()
    if game.RL:
        agent = RL_agent(action_spec=5)
        game.Rl_setup(agent.env_requirment)
        game.Rl(agent)
    if game.RL_pre:
        agent = RL_agent(action_spec=5, Pre_trained=True)
        game.Rl_setup(agent.env_requirment)
        game.Rl(agent)

    pygame.quit()
