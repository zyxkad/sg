# Copyright (C) 2023 zyxkad@gmail.com

from __future__ import annotations

import abc
import bisect
import math
import os
import random
import sys
import threading
import time
from typing import Any, Self

import pygame
import numpy
import tensorflow # type: ignore
from tensorflow import keras
from tensorflow.keras import layers # type: ignore

from ag import ( # type: ignore
	Event, on, CustomEvent, QuitBehavior,
	Director, Node, Layer, UILayer, Scene,
	Color, Colors, Vec2, Rect, Anchor,
	Surface, Font, Texture,
	Button)
from .model import ModelCli

__all__ = [
	'main'
]

ASSETS_PATH = os.path.join(os.path.dirname(__file__), 'assets')

def random_chose(weights: list[tuple[Any, float]]) -> Any:
	assert len(weights) > 0
	total_weight = sum(w for _, w in weights)
	r = random.random() * total_weight
	i = 0.0
	for x, w in weights:
		i += w
		if r < i:
			return x
	raise RuntimeError('Unexpect statment')

class Food(Node):
	_cache: dict[tuple[float, Color], Food] = {}
	score_weights = [
		(1, 10.0),
		(2, 8.0),
		(3, 3.0),
		(4, 2.0),
		(5, 1.0),
	]

	def __init__(self, x: float, y: float, size: float, score: int, color: Color, isdeadbody: bool = False):
		super().__init__(x=x, y=y, width=size, height=size)
		self.isize = size
		self._score = score
		self.color = color
		self.isdeadbody = isdeadbody

	@classmethod
	def random(cls, width: float, height: float, color: Color):
		score = random_chose(cls.score_weights)
		size = round(10 * math.sqrt(score))
		return cls(
			round((random.random() - 0.5) * (width - size)),
			round((random.random() - 0.5) * (height - size)),
			size, score, color)

	@property
	def score(self) -> int:
		return self._score

	def on_draw(self, surface: Surface):
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

class SnakeNode(Node):
	_cache: dict[tuple[int, Color], Surface] = {}
	def __init__(self, snake: Snake, angle: float, x: float, y: float):
		super().__init__(x=x, y=y, rotation=angle)
		self._snake = snake

	@property
	def snake(self) -> Snake:
		return self._snake

	@property
	def isize(self) -> int:
		return self.snake.isize

	@property
	def color(self) -> Color:
		return self.snake.body_color

	@property
	def width(self) -> float:
		return self.isize

	@property
	def height(self) -> float:
		return self.isize

	@property
	def angle(self) -> float:
		return self.rotation

	def copy(self) -> SnakeNode:
		return SnakeNode(self.snake, self.angle, self.x, self.y)

	@classmethod
	def _get_body(cls, size: int, color: Color):
		s = cls._cache.get((size, color), None)
		if s is None:
			r = size / 2
			s = Surface((size, size))
			s.circle(color, (r, r), r)
			cls._cache[(size, color)] = s
		return s

	def on_draw(self, surface: Surface):
		cls = self.__class__
		surface.clone_from(cls._get_body(self.isize, self.color))

class Snake(abc.ABC, Node):
	turn_speed = 200
	init_speed = 80
	init_nodes = 5
	speed_up_cost = 2

	def __init__(self, x: float, y: float, angle: float, name: str, body_color: Color):
		cls = self.__class__
		super().__init__()
		self._size = 16
		self.hx = x
		self.hy = y
		self.speed = cls.init_speed
		self.angle = angle % 360
		self._score = 0.0
		self.kill_count = 0
		self._last_score = 0
		self._last_score2 = 50
		self._name = name
		self.body_color = body_color
		self._dead = False
		self._nodes = [
			SnakeNode(self, self.angle, self.hx, self.hy)
				for i in range(self.node_step * cls.init_nodes)]

	@property
	def isize(self) -> int:
		return self._size

	@isize.setter
	def isize(self, size: int):
		self._size = size

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

	@property
	def dead(self) -> bool:
		return self._dead

	@property
	def children(self) -> list[Node]:
		return self._nodes

	@property
	def children_len(self) -> int:
		return len(self._nodes)

	@property
	def nodes(self) -> list[SnakeNode]:
		return self._nodes

	def kill(self):
		self._dead = True

	def reset(self, x: float, y: float, angle: float) -> Self:
		cls = self.__class__
		self.isize = 16
		self.hx = x
		self.hy = y
		self.speed = cls.init_speed
		self.angle = angle % 360
		self._score = 0.0
		self.kill_count = 0
		self._last_score = 0
		self._last_score2 = 50
		self._nodes = [
			SnakeNode(self, self.angle, self.hx, self.hy)
				for i in range(self.node_step * cls.init_nodes)]
		self._dead = False
		return self

	def _check_out_of_bounds(self, width: float, height: float) -> bool:
		return \
			self.hx < (-width + self.isize) / 2 or \
			self.hx > (width - self.isize) / 2 or \
			self.hy < (-height + self.isize) / 2 or \
			self.hy > (height - self.isize) / 2

	def move(self, dx: float, dy: float):
		self.hx += dx
		self.hy += dy

	def on_update(self, dt: float):
		width, height = self.parent.width, self.parent.height
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
			self._nodes.insert(0, SnakeNode(self, self.angle, self.hx, self.hy))
			self._nodes.pop(-1)
			if self._check_out_of_bounds(width, height):
				self._dead = True
				return

		half_size = self.isize // 2
		if self._last_score + half_size < self.score:
			self._last_score += half_size
			if self.isize <= 50 and self._last_score2 * 2 < self.score:
				self._last_score2 = self._last_score2 * 2
				self.isize += 2
				self.speed += 2
			else:
				last_node = self._nodes[-1]
				self._nodes.extend(last_node.copy() for i in range(self.node_step))
		elif self._last_score - half_size > self.score:
			self._last_score -= half_size
			if self._last_score2 > 50 and self._last_score2 // 2 > self.score:
				self._last_score2 = self._last_score2 // 2
				self.isize -= 2
				self.speed -= 2
			else:
				self._nodes = self._nodes[:-self.node_step]

	def on_draw(self, surface: Surface):
		r = self.isize / 2

		for i, n in enumerate(self.nodes):
			n.visible = i % self.node_step == 0

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

	def move(self, dx: float, dy: float):
		super().move(dx, dy)
		if abs(self.hx - self.d.camera.x) >= self.camera_follow_x:
			self.d.camera.x += dx
		if abs(self.hy - self.d.camera.y) >= self.camera_follow_y:
			self.d.camera.y += dy

	def reset(self, x: float, y: float, angle: float) -> Self:
		self.d.camera.x = x
		self.d.camera.y = y
		return super().reset(x, y, angle)

class ExcThread(threading.Thread):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._exc = None
		self._invoke_excepthook = self.__class__.excepthook

	def excepthook(self):
		self._exc = sys.exception()

	@property
	def exception(self) -> BaseException | None:
		return self._exc

	def check_exc(self) -> None:
		if not self.is_alive():
			if self.exception is not None:
				raise self.exception

class BotSnake(Snake):
	model = ModelCli()

	def __init__(self, x: int, y: int, angle: int, name: str, body_color: Color):
		super().__init__(x, y, angle, name, body_color)
		# self.model = SnakeModel(start=True)
		self._left_active = False
		self._right_active = False
		self._speed_active = False

		self._last_out = None
		self._predicting: bool | ExcThread = False
		self._last_kill_count = 0
		self._last_gain_score = 0
		self._last_action = 0
		self._survive_time = 0
		self._predict_dt = 0
		self._gradienting: bool | ExcThread = False

	@property
	def left_active(self) -> bool:
		return self._left_active

	@property
	def right_active(self) -> bool:
		return self._right_active

	@property
	def speed_active(self) -> bool:
		return self._speed_active

	def check_predicting(self) -> bool:
		if self._predicting is False:
			return False
		if self._predicting is True:
			return True
		self._predicting.check_exc()
		return True

	def check_gradienting(self) -> bool:
		if self._gradienting is False:
			return False
		if self._gradienting is True:
			return True
		self._gradienting.check_exc()
		return True

	def reset(self, x: float, y: float, angle: float) -> Self:
		super().reset(x, y, angle)
		self._left_active = False
		self._right_active = False
		self._speed_active = False

		self._last_out = None
		self._predicting = False
		self._last_kill_count = 0
		self._last_gain_score = 0
		self._last_action = 0
		self._survive_time = 0
		self._predict_dt = 0
		self._gradienting = False
		return self

	def _apply_gradients(self):
		assert not self.check_gradienting()
		self._gradienting = True
		print('[DBUG]: --> applying gradients for snake', self.name)
		def _apply_cb():
			self.model.apply_gradients()
			self._gradienting = False
		self._gradienting = ExcThread(target=_apply_cb, name='BotSnake-apply_gradients')
		self._gradienting.start()

	def _set_action(self, action: int):
		self._speed_active = bool(action & 1)
		self._left_active = bool(action & 2)
		self._right_active = bool(action & 4)
		if self._last_action != action >> 1:
			self._last_action = action >> 1
			return True
		return False

	def _set_model_out(self, out):
		self._last_out = out
		self._predicting = False

	def _predict_next_action(self, snodes: list, sfoods: list, training: bool = False):
		if self.check_predicting():
			return False
		self._predicting = True
		inl = [
			int(self.left_active), int(self.right_active), int(self.speed_active),
			self._survive_time,
			self.hx, self.hy, self.parent.width, self.parent.height, (self.angle + 360) % 360 / 90,
			self.speed, self.isize, len(self.nodes), self._score, self.kill_count]
		for i in range(0, len(self.nodes), self.node_step):
			if i > 30 * self.node_step:
				break
			n = self.nodes[i]
			inl.extend((n.x - self.hx, n.y - self.hy, (n.angle - self.angle) % 360 / 90))
		inl.extend([0] * 3 * max(0, 30 - (len(self.nodes) // self.node_step)))
		for o in snodes[:50]:
			inl.extend(o[1:])
		inl.extend([0] * 6 * max(0, 50 - len(snodes)))
		for o in sfoods[:20]:
			inl.extend(o[1:])
		inl.extend([0] * 4 * max(0, 20 - len(sfoods)))
		inl = [n / 30 for n in inl]
		self._predicting = ExcThread(
			target=(self.model.train if training else self.model.predict),
			args=([inl], self._set_model_out),
			name='predict-thread',
			daemon=True)
		self._predicting.start()
		return True

	def on_update(self, dt: float):
		super().on_update(dt)

class PlayLayer(Layer):
	def __init__(self, bwidth: int, bheight: int, size: int):
		super().__init__()
		self.d = Director()
		self.speedX = 1 # speed up game progress, for traning model (1 is the normal speed)
		self.bwidth = bwidth
		self.bheight = bheight
		self.isize = size
		self.broad_color = Color.from_rgb(0.9, 0.9, 0.9)
		self.broad_line_color = Color.from_rgb(0.5, 0.5, 0.5)

		self.snakes: list[Snake] = []
		self._player: Player | None = None
		self.training: list[Snake] = []
		self._watching = None #'Bot_0'

		self.foods: list[Food] = []

	@on('load')
	def __on_load(self):
		self.snakes = []
		self.training = []
		self.foods = []
		rp = self.random_pos()
		self._player = Player(int(rp.x), int(rp.y), int(random.random() * 4) * 90, 'Player', Colors.blue)
		self.add_snake(self.player)
		for i in range(10):
			rp = self.random_pos()
			s = BotSnake(
				int(rp.x), int(rp.y),
				int(random.random() * 360),
				f'Bot_{i}',
				Color(random.randint(0, 0xff), random.randint(0, 0xff), random.randint(0, 0xff)))
			self.training.append(s)
			self.add_snake(s)
		init_food_count = self.bwidth * self.bheight // 10
		for _ in range(init_food_count):
			self.make_food()

	@property
	def width(self) -> float:
		return self.bwidth * self.isize

	@property
	def height(self) -> float:
		return self.bheight * self.isize

	@property
	def player(self) -> Player:
		assert self._player is not None
		return self._player

	def random_pos(self) -> Vec2:
		return Vec2(
			(random.random() - 0.5) * (self.width - self.isize * 10),
			(random.random() - 0.5) * (self.height - self.isize * 10))

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
			if isinstance(s, BotSnake) and s.check_gradienting():
				self.scheduler.add_timeout(retrieve, 0.5)
				return
			print('[DBUG]: retrieving snake', s.name)
			rp = self.random_pos()
			self.add_snake(s.reset(
				int(rp.x), int(rp.y),
				int(random.random() * 360)))
		self.scheduler.add_timeout(retrieve, 3)

	def on_update(self, dt: float):
		dt = dt * self.speedX

		snakes = self.snakes.copy()
		for s in snakes:
			isbot = isinstance(s, BotSnake)
			iswatching = self._watching == s.name
			if iswatching:
				self.d.camera.x = s.hx
				self.d.camera.y = s.hy
			last_pos = s.hpos
			if isbot:
				score = 0
				if s._last_out is not None:
					out = s._last_out
					s._last_out = None
					if iswatching:
						print('[DBUG]: action props:', ', '.join(f'{p:.04f}' for p in out.action_probs))
					if s._set_action(out.action):
						score += 1
				else:
					out = None
			s.on_update(dt)
			spos = s.hpos
			dead = s.dead
			if isbot:
				if not dead:
					s._survive_time += dt
				s._last_gain_score += dt
				score += 1 if s._last_gain_score < 10 else 0
				if s.speed_active:
					score -= s.speed_up_cost * 2
				sfoods: list = []
				snodes: list = []
			for f in self.foods.copy():
				dis = spos.distance_to2(f.pos)
				if dis <= ((self.isize + f.isize) // 2) ** 2:
					s._score += f.score
					if isbot:
						score += f.score * 2
						s._last_gain_score = 0
					self.remove_food(f)
				elif isbot and dis <= (s.isize * 20) ** 2:
					bisect.insort(sfoods, (dis, (f.x - s.hx), (f.y - s.hy), f.isize, 1 if f.isdeadbody else 0), key=lambda n: n[0])
			for s2 in snakes:
				if dead:
					break
				if s2 is not s:
					for i, n in enumerate(s2.nodes[:-s2.node_step]):
						dis = spos.distance_to2(n.pos)
						if dis <= ((s.isize + s2.isize) // 2) ** 2:
							s2.kill_count += 1
							s.kill()
							dead = True
							break
						if isbot and dis <= (s.isize * 20) ** 2:
							if i % s2.node_step == 0:
								bisect.insort(snodes, (dis, (n.x - s.hx), (n.y - s.hy),
									1 if i == 0 else 0, s2.speed, s2.isize, n.angle), key=lambda n: n[0])

			if isbot:
				if out:
					if s._last_gain_score > 30: # too long idle
						print('[DBUG]: -> killed snake due too long with idling')
						s.kill()
						dead = True
					if dead:
						score = -20
					# else:
					# 	score += s.score / 5
					new_killed = s.kill_count - s._last_kill_count
					s._last_kill_count = s.kill_count
					score += new_killed * 5
					if iswatching:
						print(f'[DBUG]: score: {score:04.04f} {out.critic:04.04f}', s._survive_time)
					out.set_reward(score)
					if dead and iswatching:
						s._apply_gradients()
				if not dead:
					s._predict_dt += dt
					if s._predict_dt >= 0.2:
						if s._predict_next_action(snodes, sfoods, training=iswatching):
							s._predict_dt = 0
			if dead:
				score_remain = s.score + 10
				node_per_score = 10
				avg_score = score_remain * node_per_score // len(s.nodes)

				for n in (s.nodes[i] for i in range(0, len(s.nodes), node_per_score)):
					self.add_food(Food(
						n.x + (random.random() - 0.5) * 10,
						n.y + (random.random() - 0.5) * 10,
						max(math.sqrt(avg_score), s.isize), avg_score, s.body_color,
						isdeadbody=True))
				self.remove_snake(s)
				if s is self.player:
					self.on_player_dead()
				else:
					self.on_bot_dead(s)

	def on_draw(self, surface: Surface):
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

	def on_draw(self, surface: Surface):
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

class PauseLayer(UILayer):
	def __init__(self):
		super().__init__()
		self.d = Director()
		self.font23 = Font.default(23)
		self.bgcolor = Color(0x00, 0x00, 0x00, 0x01)

	@on('pause')
	def pause(self):
		self.visible = True

	@on('resume')
	def resume(self):
		self.visible = False

	def on_draw(self, surface: Surface):
		surface.fill(self.bgcolor)

class PlayScene(Scene):
	def __init__(self):
		super().__init__()
		self.plyl = PlayLayer(50, 50, 20)
		self.gui = GUILayer()
		self.pausel = PauseLayer()
		self.wall_color = Color(0xff, 0x40, 0x50)
		self.add_child(self.plyl)
		self.add_child(self.gui)
		self.add_child(self.pausel)

		self.schedule_update()

	def on_update(self, dt: float):
		self.gui.score = self.plyl.player.score
		self.gui.killed = self.plyl.player.kill_count
		self.plyl.on_update(dt)

		self.gui.ranks = [('<unknown>', 0, 0) for _ in range(5)]
		for i, s in enumerate(sorted(self.plyl.snakes, key=lambda s: s.score, reverse=True)):
			if i >= len(self.gui.ranks):
				break
			self.gui.ranks[i] = (s.name, s.score, s.kill_count)

	def on_draw(self, surface: Surface):
		surface.fill(self.wall_color)

class MenuLayer(UILayer):
	def __init__(self):
		super().__init__()
		self.d = Director()
		self.start_btn = Button(self.start_game,
			x=self.d.winsize.x / 2, y=self.d.winsize.y / 2, width=250, height=100,
			idle_texture=Texture(os.path.join(ASSETS_PATH, 'button/start.png')),
			hover_texture=Texture(os.path.join(ASSETS_PATH, 'button/start-hover.png')),
			click_texture=Texture(os.path.join(ASSETS_PATH, 'button/start-click.png')),
			disable_texture=Texture(os.path.join(ASSETS_PATH, 'button/start-disabled.png')),
		)
		self.add_child(self.start_btn)

	@on('load')
	def __on_load(self):
		self.start_btn.disabled = False

	def start_game(self):
		print('[DBUG]: Game starting')
		self.start_btn.disabled = True
		self.d.dispatch(CustomEvent('start_play'))

	def on_draw(self, surface: Surface):
		pass

class MenuScene(Scene):
	def __init__(self):
		super().__init__()
		self.menu = MenuLayer()
		self.add_child(self.menu)

def main():
	d = Director(quit_behavior=QuitBehavior.DESTROY_WHEN_QUIT)
	d.init_with_window((1200, 700), 'Snake Game', fps=60)
	menus = MenuScene()
	plays = PlayScene()

	@d.on('start_play')
	def _():
		d.push_scene(plays)

	@d.on('exit_play')
	def _():
		assert d.current_scene is plays
		d.pop_scene()

	d.run_with_scene(menus)

	# for s in ps.plyl.snakes:
	# 	if ps.plyl._watching == s.name:
	# 		s._apply_gradients()

	# done_flag = False
	# while not done_flag:
	# 	done_flag = True
	# 	for s in ps.plyl.training:
	# 		if s.check_gradienting():
	# 			done_flag = False
	# 			break
	# 	time.sleep(0.2)

	# BotSnake.stop()
