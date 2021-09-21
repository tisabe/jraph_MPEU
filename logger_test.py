import logger
import optax

logger = logger.Logger()
logger.print_start_time()
logger.log_str('This message was brought to you by the test!')

lr_schedule = optax.exponential_decay(1e-3, 1000, 0.9)
logger.log_hyperparams(1024, 10, 32, lr_schedule)
logger.log_loss(1e-2, 100)