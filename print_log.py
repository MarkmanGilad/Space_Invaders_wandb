from PPO_Trainer import Logger
'''
actor_loss critic_loss total_loss
actor_lr  critic_lr
score   level
max_actor_grad  max_critic_grad
advantage_mean  advantage_std
'''

logger = Logger(chkpt=51)
logger.load()
# logger.print_keys()
# logger.print_all()
# logger.print_key('advantage_mean',5)
# logger.print_key('advantage_std',5)
# logger.print_key('max_actor_grad',5)
# logger.print_key('max_critic_grad',5)
# logger.print_key('actor_lr')
# logger.print_key('critic_lr')
# logger.print_key('actor_loss',2)
# logger.print_key('critic_loss',2)
logger.print_key('advantage',5)

