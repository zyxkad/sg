
import json
import sys
import tensorflow

_ready_mark = '======== READY MARK HERE ========='

class ModelOutput:
	def __init__(self, outs):
		self.action_probs = outs[0][0]
		self.critic = outs[1][0, 0]
		self.action = numpy.random.choice(len(self.action_probs), p=numpy.squeeze(self.action_probs))
		self.reward = None
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
		self.reward = None
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
			self.p = subprocess.Popen([sys.executable, __file__],
				stdin=subprocess.PIPE, stdout=subprocess.PIPE,
				bufsize=8192, pipesize=8192)
			self._stdin = io.TextIOWrapper(self.p.stdin)
			self._stdout = io.TextIOWrapper(self.p.stdout)
			self.lock = threading.Lock()
			self.outputs = []

			while True:
				l = self._stdout.readline()
				if not l:
					raise RuntimeError('Model subprocess failed')
				if l[:-1] == _ready_mark:
					break
				time.sleep(0.2)

		def _check_alive(self):
			if self.p is None or self.p.poll() is not None:
				raise RuntimeError('Subprocess exited')

		def stop(self):
			with self.lock:
				self.p.kill()
				self.p = None

		def predict(self, data):
			with self.lock:
				self._check_alive()
				self._stdin.write('p ')
				json.dump(data, self._stdin)
				self._stdin.write('\n')
				self._stdin.flush()
				l = self._stdout.readline()
				self._check_alive()
				return ModelOutput.deserialize(json.loads(l))

		def train(self, data):
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
				return out

		def apply_gradients(self):
			with self.lock:
				self._check_alive()
				self._stdin.write('g ')
				json.dump([0 if o.reward is None else o.reward for o in self.outputs], self._stdin)
				self._stdin.write('\n')
				self._stdin.flush()
				l = self._stdout.readline()
				self.outputs.clear()
				self._check_alive()
				return json.loads(l)

else:
	import os
	import numpy
	from tensorflow import keras
	from tensorflow.keras import layers

	outputfd = sys.stdout
	sys.stdout = sys.stderr
	use_gpu = True

	EPS = numpy.finfo(numpy.float32).eps.item()

	def newSnakeModel():
		size = 3 + 8 + 6 * 50 + 4 * 20
		inputs = keras.Input(shape=(size,))
		d1 = layers.Dense(size, activation=layers.LeakyReLU(alpha=0.1))(inputs)
		d2 = layers.Dense(97, activation='sigmoid')(d1)
		# 6 status [left keep right] * [slow fast]
		action = layers.Dense(6, activation='softmax')(d2)
		critic = layers.Dense(1)(d2)
		return keras.Model(inputs=inputs, outputs=[action, critic])


	def calc_discounted(rewards, gamma=0.99):
		discounted_sum = 0
		returns = []
		for r in reversed(rewards):
			discounted_sum = r + gamma * discounted_sum
			returns.append(discounted_sum)
		returns = numpy.array(list(reversed(returns)))
		returns = (returns - numpy.mean(returns)) / (numpy.std(returns) + EPS).tolist()
		return returns

	model_target = './snake0.keras'
	model_saved = False
	def save_model():
		global model_saved
		if model_saved:
			return
		model_saved = True
		print('==> Saving model to', model_target)
		model.compile()
		model.save(model_target)
	if os.path.exists(model_target):
		print('==> Loading model from', model_target)
		model = keras.models.load_model(model_target)
		print('==> Model loaded')
	else:
		model = newSnakeModel()

	def main(outputfd):
		optimizer = keras.optimizers.legacy.Adam(learning_rate=0.01)
		huber_loss = keras.losses.Huber()

		tape = tensorflow.GradientTape()
		with tape:
			action_probs_history = []
			critic_value_history = []
			outputfd.write(_ready_mark)
			outputfd.write('\n')
			outputfd.flush()
			while True:
				l = sys.stdin.readline()
				if not l:
					break
				cmd, l = l.split(' ', 1)
				obj = json.loads(l)
				if cmd == 'p': # predict
					out = ModelOutput(model.predict(tensorflow.convert_to_tensor(obj, dtype=tensorflow.float32)))
					json.dump(out.serialize(), outputfd, check_circular=False)
					outputfd.write('\n')
					outputfd.flush()
				elif cmd == 't': # train
					out = ModelOutput(model(tensorflow.convert_to_tensor(obj, dtype=tensorflow.float32)))
					json.dump(out.serialize(), outputfd, check_circular=False)
					outputfd.write('\n')
					outputfd.flush()
					action_probs_history.append(tensorflow.math.log(out.action_probs[out.action]))
					critic_value_history.append(out.critic)
				elif cmd == 'g':
					returns = calc_discounted(obj)
					loss_value = None
					c_losses = []
					for a, c, r in zip(action_probs_history, critic_value_history, returns):
						l = a * (c - r)
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
					outputfd.flush()

	with tensorflow.device('/GPU' if use_gpu else '/CPU'):
		main(outputfd)
		save_model()
