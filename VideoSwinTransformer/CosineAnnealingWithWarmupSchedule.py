## Manually implemented cosine decay with warmup for using with AdamW optimizer
import numpy as np
import tensorflow as tf

def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):

    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')

    if not isinstance(global_step, int):
      global_step = 1
     
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(
        np.pi *
        (global_step - warmup_steps - hold_base_rate_steps
        ) / float(total_steps - warmup_steps - hold_base_rate_steps)))


    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)
        
    return np.where(global_step > total_steps, 0.0, learning_rate)


class CosineAnnealingWithWarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, learning_rate_base,total_steps,warmup_learning_rate=0.0,warmup_steps=0,hold_base_rate_steps=0):
    super().__init__()

    
    self.learning_rate_base = learning_rate_base
    self.total_steps = total_steps
    self.warmup_learning_rate = warmup_learning_rate
    self.warmup_steps = warmup_steps
    self.hold_base_rate_steps = hold_base_rate_steps

  def __call__(self, step):
    lr = cosine_decay_with_warmup(global_step=step,
                                      learning_rate_base=self.learning_rate_base,
                                      total_steps=self.total_steps,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      warmup_steps=self.warmup_steps,
                                      hold_base_rate_steps=self.hold_base_rate_steps)
    print("lr =", lr)
    return lr