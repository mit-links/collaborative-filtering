import datetime as dt
import json
import sys

from SVD import Parameters, Predictor, Tester


class RecommenderSystem:
    def __init__(self):
        self.config_file = self._read_config_file()

    def _read_config_file(self):
        config_file = sys.argv[1]

        with open(config_file) as config_json:
            data = json.load(config_json)
        return data

    def start(self):
        params = Parameters(self.config_file["params"])
        if self.config_file["predict_only"]:  # only predict with the first set of parameters, no testing
            p = Predictor(params.get_next_parameters())
            p.start()
        else:
            t = Tester(params)
            t.start()


if __name__ == '__main__':
    start = dt.datetime.now()

    rs = RecommenderSystem()
    rs.start()

    end = dt.datetime.now()
    diff = end - start
    print("took " + str(diff))
