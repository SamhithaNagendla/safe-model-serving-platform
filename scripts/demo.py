from pathlib import Path

from model_serving.metrics import MetricsStore
from model_serving.routing import RoutingConfig
from model_serving.service import ModelService

service = ModelService("artifacts", MetricsStore(Path("data/demo-metrics.db")))
service.update_routing(RoutingConfig(challenger_percent=25))
try:
    for index in range(1000):
        service.predict(f"user-{index}", [index % 5, 1])
    print(service.metrics.summary())
    print("rollback:", service.rollback())
finally:
    service.close()
