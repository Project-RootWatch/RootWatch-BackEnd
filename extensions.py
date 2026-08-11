from flask_sqlalchemy import SQLAlchemy
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

db = SQLAlchemy()

# In-memory storage by default — fine for a single dev process, but it
# resets on every restart and doesn't share state across multiple worker
# processes. A real production deployment (multiple Gunicorn workers, for
# instance) needs a shared store — Redis is the standard choice — or each
# worker enforces its own separate limit, effectively multiplying it.
limiter = Limiter(key_func=get_remote_address)
