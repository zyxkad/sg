# Copyright (C) 2023 zyxkad@gmail.com

import abc
import bisect
import math
import os
import random
import time
import threading
from typing import Any, Self

import pygame
import numpy
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers

from ag import *
from .model import ModelCli

__all__ = [
	'main'
]

def random_chose(weights: list[tuple[Any, float]]) -> Any:
	assert len(weights) > 0
	total_weight = sum(w for _, w in weights)
	r = random.random() * total_weight
	i = 0
	for x, w in weights:
		i += w
		if r < i:
			return x
	raise RuntimeError('Unexpect statment')

class Food(Node):
	_cache = {}
	score_weights = [
		(1, 10.0),
		(2, 8.0),
		(3, 3.0),
		(4, 2.0),
		(5, 1.0),
	]

	def __init__(self, x: int, y: int, size: int, score: int, color: Color, isdeadbody: bool = False):
		super().__init__(x=x, y=y, width=size, height=size)
		self.isize = size
		self._score = score
		self.color = color
		self.isdeadbody = isdeadbody

	@classmethod
	def random(cls, width: int, height: int, color: Color):
		score = random_chose(cls.score_weights)
		size = round(10 * math.sqrt(score))
		return cls(
			round((random.random() - 0.5) * (width - size)),
			round((random.random() - 0.5) * (height - size)),
			size, score, color)

	@property
	def score(self) -> int:
		return self._score

	def on_draw(self, surface):
		cls = self.__class__
		ch = cls._cache.get((self.isize, self.color), None)
		if ch is None:
			ch = Surface((self.width, self.height))
			ch.polygon(self.color, [
				Vec2(self.isize / 4, self.isize / 2),
				Vec2(self.isize / 2, 0),
				Vec2(self.isize / 4 * 3, self.isize / 2),
				Vec2(self.isize / 2, self.isize),
			])
			cls._cache[(self.isize, self.color)] = ch
		surface.clone_from(ch)

class Snake(abc.ABC, Node):
	turn_speed = 100
	init_speed = 50
	init_nodes = 5
	speed_up_cost = 2

	def __init__(self, x: int, y: int, angle: int, name: str, body_color: Color):
		cls = self.__class__
		super().__init__()
		self.isize = 16
		self.hx = x
		self.hy = y
		self.speed = cls.init_speed
		self.angle = angle % 360
		self._score = 0
		self.kill_count = 0
		self._last_score = 0
		self._last_score2 = 50
		self.nodes = [(self.hx, self.hy, self.angle) for i in range(self.node_step * cls.init_nodes)]
		self._name = name
		self.body_color = body_color
		self._dead = False

	@property
	def hpos(self) -> Vec2:
		return Vec2(self.hx, self.hy)

	@property
	def score(self) -> int:
		return int(self._score)

	@property
	def node_step(self) -> int:
		return int(self.isize // 2 * 100 // self.speed * 0.85)

	@property
	def name(self) -> str:
		return self._name

	@property
	@abc.abstractmethod
	def left_active(self) -> bool:
		raise NotImplementedError()

	@property
	@abc.abstractmethod
	def right_active(self) -> bool:
		raise NotImplementedError()

	@property
	@abc.abstractmethod
	def speed_active(self) -> bool:
		raise NotImplementedError()

	def reset(self, x: int, y: int, angle: int) -> Self:
		cls = self.__class__
		self.isize = 16
		self.hx = x
		self.hy = y
		self.speed = cls.init_speed
		self.angle = angle % 360
		self._score = 0
		self.kill_count = 0
		self._last_score = 0
		self._last_score2 = 50
		self.nodes = [(self.hx, self.hy, self.angle) for i in range(self.node_step * cls.init_nodes)]
		self._dead = False
		return self

	def move(self, dx: float, dy: float):
		self.hx += dx
		self.hy += dy

	def on_update(self, dt: float):
		speed = 1
		if self.speed_active:
			if self._score >= self.speed_up_cost * dt:
				self._score -= self.speed_up_cost * dt
				speed = 2
		dtf = 0.02
		dtn = int(dt / dtf)
		step = self.speed * dtf
		for _ in range(dtn * speed):
			if self.left_active:
				self.angle = (self.angle + self.turn_speed * dtf) % 360
			if self.right_active:
				self.angle = (self.angle - self.turn_speed * dtf) % 360
			dx = -step * math.cos(self.angle / 180 * math.pi)
			dy = step * math.sin(self.angle / 180 * math.pi)
			self.move(dx, dy)
			self.nodes.insert(0, (self.hx, self.hy, self.angle))
			self.nodes.pop(-1)

		half_size = self.isize // 2
		if self._last_score + half_size < self.score:
			self._last_score += half_size
			if self.isize <= 50 and self._last_score2 * 2 < self.score:
				self._last_score2 = self._last_score2 * 2
				self.isize += 2
				self.speed += 2
			else:
				last_node = self.nodes[-1]
				self.nodes.extend(last_node for i in range(self.node_step))
		elif self._last_score - half_size > self.score:
			self._last_score -= half_size
			if self._last_score2 > 50 and self._last_score2 // 2 > self.score:
				self._last_score2 = self._last_score2 // 2
				self.isize -= 2
				self.speed -= 2
			else:
				self.nodes = self.nodes[:-self.node_step]

	def on_draw(self, surface):
		r = self.isize / 2
		head = Surface((self.isize, self.isize))
		head.circle(self.body_color, (r, r), r)
		surface.blit(head, (self.hx, self.hy))

		body = Surface((self.isize, self.isize))
		body.circle(self.body_color, (r, r), r)
		for i in range(self.node_step, len(self.nodes), self.node_step):
			x, y, a = self.nodes[i]
			surface.blit(body, (x, y))

		name_sf = Font.default(20).render(self.name, Colors.white, True)
		namebg_sf = Surface(name_sf.size + (6, 4))
		namebg_sf.fill(Color(0x00, 0x00, 0x00, 0x40))
		namebg_sf.blit(name_sf, (3, 3), anchor=Anchor.TOP_LEFT)
		surface.blit(namebg_sf, (self.hx, self.hy - 6), anchor=Anchor.BOTTOM_CENTER)

class Player(Snake):
	def __init__(self, x: int, y: int, angle: int, name: str, body_color: Color):
		super().__init__(x, y, angle, name, body_color)
		self.d = Director()
		self.d.camera.x = x
		self.d.camera.y = y
		self.camera_follow_x = 100
		self.camera_follow_y = 60

	@property
	def left_active(self) -> bool:
		return self.d.is_keydown(pygame.constants.K_a)

	@property
	def right_active(self) -> bool:
		return self.d.is_keydown(pygame.constants.K_d)

	@property
	def speed_active(self) -> bool:
		return self.d.is_keydown(pygame.constants.K_SPACE)

	def move(self, dx: int, dy: int):
		super().move(dx, dy)
		if abs(self.hx - self.d.camera.x) >= self.camera_follow_x:
			self.d.camera.x += dx
		if abs(self.hy - self.d.camera.y) >= self.camera_follow_y:
			self.d.camera.y += dy

	def reset(self, x: int, y: int, angle: int) -> Self:
		self.d.camera.x = x
		self.d.camera.y = y
		return super().reset(x, y, angle)

class BotSnake(Snake):
	model = ModelCli()

	def __init__(self, x: int, y: int, angle: int, name: str, body_color: Color):
		super().__init__(x, y, angle, name, body_color)
		# self.model = SnakeModel(start=True)
		self._left_active = False
		self._right_active = False
		self._speed_active = False
		self._last_kill_count = 0
		self._last_gain_score = time.time()
		self._last_action = 0
		self._training = False

	@property
	def left_active(self) -> bool:
		return self._left_active

	@property
	def right_active(self) -> bool:
		return self._right_active

	@property
	def speed_active(self) -> bool:
		return self._speed_active

	def reset(self, x: int, y: int, angle: int) -> Self:
		super().reset(x, y, angle)
		self._left_active = False
		self._right_active = False
		self._speed_active = False
		self._last_kill_count = 0
		self._last_gain_score = time.time()
		self._last_action = 0
		self._training = False
		return self

	def _apply_gradients(self):
		assert not self._training
		print('--> applying gradients for snake', self.name)
		self._training = True
		def _apply_cb():
			self.model.apply_gradients()
			self._training = False
		threading.Thread(target=_apply_cb, name='BotSnake-apply_gradients').start()

	def _set_action(self, action: int):
		self._speed_active = bool(action & 1)
		self._left_active = bool(action & 2)
		self._right_active = bool(action & 4)

class PlayLayer(Layer):
	def __init__(self, bwidth: int, bheight: int, size: int):
		super().__init__()
		self.d = Director()
		self.bwidth = bwidth
		self.bheight = bheight
		self.isize = size
		self.broad_color = Color.from_rgb(0.9, 0.9, 0.9)
		self.broad_line_color = Color.from_rgb(0.5, 0.5, 0.5)

		self.snakes = []
		self.training = []
		rp = self.random_pos()
		self.player = Player(int(rp.x), int(rp.y), int(random.random() * 4) * 90, 'Player', Colors.blue)
		# self.add_snake(self.player)
		for i in range(1):
			rp = self.random_pos()
			s = BotSnake(
				int(rp.x), int(rp.y),
				int(random.random() * 360),
				f'Bot_{i}',
				Color(random.randint(0, 0xff), random.randint(0, 0xff), random.randint(0, 0xff)))
			self.training.append(s)
			self.add_snake(s)
		self._watching = 'Bot_0'

		self.foods = []
		init_food_count = self.bwidth * self.bheight // 10
		for _ in range(init_food_count):
			self.make_food()

	@property
	def width(self) -> float:
		return self.bwidth * self.isize

	@property
	def height(self) -> float:
		return self.bheight * self.isize

	def random_pos(self) -> Vec2:
		return Vec2(
			(random.random() - 0.5) * (self.width - self.isize * 6),
			(random.random() - 0.5) * (self.height - self.isize * 6))

	def add_snake(self, s: Snake):
		self.snakes.append(s)
		self.add_child(s, z_index=10)

	def remove_snake(self, s: Snake):
		self.snakes.remove(s)
		self.remove_child(s)

	def add_food(self, f: Food):
		self.add_child(f)
		self.foods.append(f)

	def remove_food(self, f: Food):
		self.foods.remove(f)
		self.remove_child(f)
		if not f.isdeadbody:
			self.make_food()

	def on_player_dead(self):
		def retrieve():
			rp = self.random_pos()
			self.add_snake(self.player.reset(
				int(rp.x), int(rp.y),
				int(random.random() * 8) * 45))
		self.scheduler.add_timeout(retrieve, 3)

	def on_bot_dead(self, s: Snake):
		def retrieve():
			if isinstance(s, BotSnake) and s._training:
				self.scheduler.add_timeout(retrieve, 0.5)
				return
			print('retrieving snake', s.name)
			rp = self.random_pos()
			self.add_snake(s.reset(
				int(rp.x), int(rp.y),
				int(random.random() * 360)))
		self.scheduler.add_timeout(retrieve, 3)

	def on_entered(self):
		super().on_entered()

	def on_exit(self):
		super().on_entered()

	def on_update(self, dt: float):
		snakes = self.snakes.copy()
		for s in snakes:
			isbot = isinstance(s, BotSnake)
			if self._watching == s.name:
				self.d.camera.x = s.hx
				self.d.camera.y = s.hy
			last_pos = s.hpos
			s.on_update(dt)
			spos = s.hpos
			if isbot:
				now = time.time()
				score = (-dt if s.speed_active else 0) - (now - s._last_gain_score) ** 2 / 10 + 1
				sfoods = []
				snodes = []
			for f in self.foods.copy():
				dis = spos.distance_to2(f.pos)
				if dis <= ((self.isize + f.isize) // 2) ** 2:
					s._score += f.score
					if isbot:
						score += f.score
						s._last_gain_score = now
					self.remove_food(f)
				elif isbot and dis <= (s.isize * 20) ** 2:
					bisect.insort(sfoods, (dis, (f.x - s.hx) / self.width, (f.y - s.hy) / self.height, f.isize, 1 if f.isdeadbody else 0), key=lambda n: n[0])
			dead = \
				s.hx < (-self.width + s.isize) // 2 or \
				s.hx > (self.width - s.isize) // 2 or \
				s.hy < (-self.height + s.isize) // 2 or \
				s.hy > (self.height - s.isize) // 2
			for s2 in snakes:
				if dead:
					break
				if s2 is not s:
					for i, (x, y, a) in enumerate(s2.nodes[:-s2.node_step]):
						p = Vec2(x, y)
						dis = spos.distance_to2(p)
						if dis <= ((s.isize + s2.isize) // 2) ** 2:
							s2.kill_count += 1
							dead = True
							break
						if isbot and dis <= (s.isize * 20) ** 2:
							if i % s2.node_step == 0:
								bisect.insort(snodes, (dis, (x - s.hx) / self.width, (y - s.hy) / self.height, 1 if i == 0 else 0, s2.speed, s2.isize, a), key=lambda n: n[0])

			if isbot:
				inl = [
					int(s.left_active), int(s.right_active), int(s.speed_active),
					s.hx / self.width, s.hy / self.height, (s.angle + 360) % 360 / 360,
					s.speed, s.isize, len(s.nodes), s._score, s.kill_count]
				for o in snodes[:50]:
					inl.extend(o[1:])
				inl.extend([0] * 6 * max(0, 50 - len(snodes)))
				for o in sfoods[:20]:
					inl.extend(o[1:])
				inl.extend([0] * 4 * max(0, 20 - len(sfoods)))
				out = s.model.train([inl])
				s._set_action(out.action)
				if now > s._last_gain_score + 20: # too long idle
					print('killed snake due too long with idling')
					dead = True
				if dead:
					s._apply_gradients()
					score -= s.score + 100
				else:
					score += s.score / 10
				new_killed = s.kill_count - s._last_kill_count
				s._last_kill_count = s.kill_count
				score += new_killed * 5
				# if self._watching == s.name:
				# 	print('score:', score)
				out.set_reward(score)
			if dead:
				score_remain = s.score + 10
				node_per_score = 10
				avg_score = score_remain * node_per_score // len(s.nodes)

				for x, y, _ in (s.nodes[i] for i in range(0, len(s.nodes), node_per_score)):
					self.add_food(Food(
						x + (random.random() - 0.5) * 10,
						y + (random.random() - 0.5) * 10,
						max(math.sqrt(avg_score), s.isize), avg_score, s.body_color,
						isdeadbody=True))
				self.remove_snake(s)
				if s is self.player:
					self.on_player_dead()
				else:
					self.on_bot_dead(s)

	def on_draw(self, surface):
		surface.fill(self.broad_color)
		for x in range(1, self.bwidth):
			surface.fill(self.broad_line_color, Rect(x * self.isize, 0, 1, self.height))
		for y in range(1, self.bheight):
			surface.fill(self.broad_line_color, Rect(0, y * self.isize, self.width, 1))

	def make_food(self):
		f = Food.random(self.width, self.height, Colors.yellow)
		self.add_food(f)

class GUILayer(UILayer):
	def __init__(self):
		super().__init__()
		self.d = Director()
		self.font = Font.default(20)
		self.score = 0
		self.killed = 0
		self.ranks = []

	def on_draw(self, surface):
		fps = self.d.real_fps
		left_box = Surface((100, 70))
		left_box.fill(Color(0, 0, 0, 0x50))
		left_box.blit(
			self.font.render(f'FPS: {fps:.2f}', Colors.white, True),
			Vec2(5, 5), anchor=Anchor.TOP_LEFT)
		left_box.blit(
			self.font.render(f'Score: {self.score:d}', Colors.white, True),
			Vec2(5, 25), anchor=Anchor.TOP_LEFT)
		left_box.blit(
			self.font.render(f'Killed: {self.killed:d}', Colors.white, True),
			Vec2(5, 45), anchor=Anchor.TOP_LEFT)
		surface.blit(left_box, (0, 0), anchor=Anchor.TOP_LEFT)

		right_box = Surface((200, len(self.ranks) * 20 + 5))
		right_box.fill(Color(0, 0, 0, 0x80))
		for i, (n, s, k) in enumerate(self.ranks):
			right_box.blit(
				self.font.render(f'{i + 1} {n:12s} {s:-7d} {k:-4d}', Colors.white, True),
				Vec2(5, 5 + i * 20), anchor=Anchor.TOP_LEFT)
		surface.blit(right_box, (surface.size.x, 0), anchor=Anchor.TOP_RIGHT)

class PlayScene(Scene):
	def __init__(self):
		plyl = PlayLayer(50, 50, 20)
		gui = GUILayer()
		super().__init__(plyl, gui)
		self.plyl = plyl
		self.gui = gui
		self.wall_color = Color(0xff, 0x40, 0x50)

	def on_entered(self):
		super().on_entered()
		self.schedule_update()

	def on_update(self, dt: float):
		self.gui.score = self.plyl.player.score
		self.gui.killed = self.plyl.player.kill_count
		self.plyl.on_update(dt)

		self.gui.ranks = [('<unknown>', 0, 0) for _ in range(5)]
		for i, s in enumerate(list(sorted(self.plyl.snakes, key=lambda s: s.score, reverse=True))[:5]):
			self.gui.ranks[i] = (s.name, s.score, s.kill_count)

	def on_draw(self, surface):
		surface.fill(self.wall_color)

def main():
	d = Director(exit_when_quit=False)
	d.init_with_window((1200, 700), 'Snake Game')
	winsize = d.winsize
	ps = PlayScene()
	d.fps = 60
	d.run_with_scene(ps)

	for s in ps.plyl.snakes:
		s._apply_gradients()

	done_flag = False
	while not done_flag:
		done_flag = True
		for s in ps.plyl.snakes:
			if s._training:
				done_flag = False
				break
		time.sleep(0.2)
