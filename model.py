# Copyright (C) 2023 zyxkad@gmail.com

import json
import random
import sys
import tensorflow # type: ignore

_READY_MARK = '======== READY MARK HERE ========='

def random_chose(weights: list[float]) -> int:
	assert len(weights) > 0
	total_weight = 0.0
	for w in weights:
		assert w >= 0, f'weight ({w}) must greather or equal than zero'
		total_weight += w
	if total_weight == 0:
		return random.randint(0, len(weights))
	r = random.random() * total_weight
	i = 0.0
	for j, w in enumerate(weights):
		i += w
		if r < i:
			return j
	raise RuntimeError('Unexpect statment')

model_size = 14 + 3 * 30 + 6 * 50 + 4 * 20

class ModelOutput:
	def __init__(self, outs):
		self.action_probs = outs[0][0]
		self.critic = outs[1][0, 0]
		self.action = random_chose([abs(float(w)) for w in self.action_probs])
		self.reward = 0.0
		self.applyed = False
		self.prev = None

	def serialize(self) -> dict:
		return {
			'action_probs': self.action_probs.numpy().tolist(),
			'critic': float(self.critic),
			'action': self.action,
		}

	@classmethod
	def deserialize(cls, obj: dict):
		self = cls.__new__(cls)
		self.action_probs = tensorflow.convert_to_tensor(obj['action_probs'], dtype=tensorflow.float32)
		self.critic = tensorflow.convert_to_tensor(obj['critic'], dtype=tensorflow.float32)
		self.action = obj['action']
		self.reward = 0.0
		self.applyed = False
		self.prev = None
		return self

	def set_reward(self, reward: float):
		self.reward = reward

if __name__ != '__main__': # when import as a model
	import io
	import subprocess
	import threading
	import time

	__all__ = [
		'ModelCli'
	]

	class ModelCli:
		def __init__(self):
			self.p: subprocess.Popen | None = subprocess.Popen([sys.executable, __file__],
				stdin=subprocess.PIPE, stdout=subprocess.PIPE,
				bufsize=8192, pipesize=8192)
			assert self.p.stdin is not None
			assert self.p.stdout is not None
			self._stdin = io.TextIOWrapper(self.p.stdin)
			self._stdout = io.TextIOWrapper(self.p.stdout)
			self.lock = threading.Lock()
			self.outputs = []

			while True:
				l = self._stdout.readline()
				if not l:
					raise RuntimeError('Model subprocess failed')
				if l[:-1] == _READY_MARK:
					break
				time.sleep(0.2)

		def _check_alive(self):
			assert self.p is not None, 'Subprocess already exited'
			if self.p.poll() is not None:
				raise RuntimeError('Subprocess exited unexpectly: {}'.format(self.p.returncode))

		def stop(self):
			with self.lock:
				if self.p is not None:
					self._stdin.write('e\n')
					self._stdin.flush()
					self.p.wait()
					self.p = None

		def predict(self, data, callback=None):
			assert len(data[0]) == model_size, f'Unexpect data size {len(data[0])}, expect {model_size}'
			with self.lock:
				self._check_alive()
				self._stdin.write('p ')
				json.dump(data, self._stdin)
				self._stdin.write('\n')
				self._stdin.flush()
				l = self._stdout.readline()
				self._check_alive()
			out = ModelOutput.deserialize(json.loads(l))
			if callback is None:
				return out
			return callback(out)

		def train(self, data, callback=None):
			assert len(data[0]) == model_size, f'Unexpect data size {len(data[0])}, expect {model_size}'
			with self.lock:
				self._check_alive()
				self._stdin.write('t ')
				json.dump(data, self._stdin)
				self._stdin.write('\n')
				self._stdin.flush()
				l = self._stdout.readline()
				self._check_alive()
				out = ModelOutput.deserialize(json.loads(l))
				self.outputs.append(out)
			if callback is None:
				return out
			return callback(out)

		def apply_gradients(self, callback=None):
			with self.lock:
				self._check_alive()
				self._stdin.write('g ')
				json.dump([o.reward for o in self.outputs], self._stdin)
				self._stdin.write('\n')
				self._stdin.flush()
				l = self._stdout.readline()
				self.outputs.clear()
				self._check_alive()
			out = json.loads(l)
			if callback is None:
				return out
			return callback(out)

else: # when run as a program
	import os
	import numpy
	from tensorflow import keras
	from tensorflow.keras import layers # type: ignore

	DEBUG = False
	if DEBUG:
		from matplotlib import pyplot # type: ignore

	outputfd = sys.stdout
	sys.stdout = sys.stderr
	use_gpu = True

	EPS = numpy.finfo(numpy.float32).eps.item()

	if DEBUG:
		pyplot.ion()
		class DebugLayer(keras.layers.Layer):
			index = 0
			enabled = True

			def __init__(self, debug_name: str | None = None, *args, **kwargs):
				super().__init__(*args, **kwargs)
				self.__class__.index += 1
				self.debug_index = self.__class__.index
				self.debug_name = debug_name or f'debug-layer {self.debug_index}'
				self.fig = pyplot.figure()
				self.fig.canvas.manager.set_window_title(self.debug_name)
				self.fig.suptitle = self.debug_name
				self.ax = self.fig.add_subplot(111)

			def call(self, inputs):
				if not self.enabled:
					return inputs
				if type(inputs) is tensorflow.Tensor:
					return inputs
				l = [float(n) for n in inputs[0].numpy()]
				self.ax.clear()
				self.ax.plot(l, '.')
				self.fig.canvas.draw()
				self.fig.canvas.flush_events()
				return inputs

	def newSnakeModel():
		inputs = keras.Input(shape=(model_size,))
		# inputs = DebugLayer('inputs')(inputs)
		d1 = layers.Dense(512, activation='relu',
			kernel_initializer='random_normal',
			bias_initializer='zeros')(inputs)
		# d1 = DebugLayer('d1')(d1)
		d2 = layers.Dense(256, activation='sigmoid')(d1)
		# d2 = DebugLayer('d2-1')(d2)
		d3 = layers.Dense(128, activation='sigmoid')(d2)
		# d3 = DebugLayer('d3')(d3)
		# 6 status [left keep right] * [slow fast]
		action = layers.Dense(6, activation='softmax')(d3)
		# action = DebugLayer('action')(action)
		critic = layers.Dense(1)(d3)
		return keras.Model(inputs=inputs, outputs=[action, critic])

	def calc_discounted(rewards, gamma=0.75):
		discounted_sum = 0
		returns = []
		for r in reversed(rewards):
			discounted_sum = r + gamma * discounted_sum
			returns.append(discounted_sum)
		nreturns = numpy.array(list(reversed(returns)))
		return (nreturns - numpy.mean(nreturns)) / (numpy.std(nreturns) + EPS).tolist()

	model_target = './snake.keras'
	model_saved = False
	def save_model():
		global model_saved
		if model_saved:
			return
		model_saved = True
		print('[DBUG]: ==> Saving model to', model_target)
		model.compile()
		model.save(model_target)
	if os.path.exists(model_target):
		print('[DBUG]: ==> Loading model from', model_target)
		model = keras.models.load_model(model_target)
		print('[DBUG]: ==> Model loaded')
	else:
		model = newSnakeModel()

	def main(outputfd):
		optimizer = keras.optimizers.legacy.Adam(learning_rate=0.01)
		huber_loss = keras.losses.Huber()

		tape = tensorflow.GradientTape()
		with tape:
			action_probs_history = []
			critic_value_history = []
			outputfd.write(_READY_MARK)
			outputfd.write('\n')
			outputfd.flush()
			while True:
				l = sys.stdin.readline()
				if not l:
					# raise EOFError()
					return False
				cmd, l = l.split(' ', 1)
				obj = json.loads(l)
				if cmd == 'e': # save and exit
					return True
				elif cmd == 'p': # predict
					with tape.stop_recording():
						out = ModelOutput(model(tensorflow.convert_to_tensor(obj, dtype=tensorflow.float32)))
						json.dump(out.serialize(), outputfd)
					outputfd.write('\n')
					outputfd.flush()
				elif cmd == 't': # train
					print('[DBUG]: max min:', max(obj[0]), min(obj[0]))
					out = ModelOutput(model(tensorflow.convert_to_tensor(obj, dtype=tensorflow.float32)))
					json.dump(out.serialize(), outputfd)
					outputfd.write('\n')
					outputfd.flush()
					action_probs_history.append(tensorflow.math.log(out.action_probs[out.action]))
					critic_value_history.append(out.critic)
				elif cmd == 'g':
					if len(obj) == 0:
						outputfd.write('true\n')
					else:
						returns = calc_discounted(obj)
						loss_value = None
						c_losses = []
						for a, c, r in zip(action_probs_history, critic_value_history, returns):
							if r is None:
								continue
							l = -a * (r - c)
							if loss_value is None:
								loss_value = l
							else:
								loss_value += l
							loss_value += huber_loss(tensorflow.expand_dims(c, 0), tensorflow.expand_dims(r, 0))
						action_probs_history.clear()
						critic_value_history.clear()
						if loss_value is None:
							outputfd.write('false\n')
						else:
							with tape.stop_recording():
								gradients = tape.gradient(loss_value, model.trainable_weights)
								optimizer.apply_gradients(zip(gradients, model.trainable_weights))
							outputfd.write('true\n')
					tape.reset()
					outputfd.flush()

	with tensorflow.device('/GPU' if use_gpu else '/CPU'):
		if main(outputfd):
			save_model()
