from gp_artemis import ExperimentManager
from comparative_evaluation import Ticker
import pandas as pd
from dateutil import parser
import datetime

from gp_utils import Period


class SlidingWindowValidator:

    def __init__(self, experiment_json_template_path):
        with open(experiment_json_template_path, 'r') as f:
            self.experiment_json_template = f.read()

    def run(self, training_period, validation_period, step, end_time_str, out_filename='sliding_window.xlsx'):
        end_time = parser.parse(end_time_str).timestamp()
        dataframes = []

        while(validation_period.end_time < end_time):
            df = self._run_experiments_and_get_results(training_period, validation_period)
            training_period = training_period.get_shifted_forward(step)
            validation_period = validation_period.get_shifted_forward(step)
            dataframes.append(df)

        df = pd.concat(dataframes)
        writer = pd.ExcelWriter(out_filename)
        df.to_excel(writer, 'Sheet1')
        writer.save()
        writer.close()


    def _run_experiments_and_get_results(self, training_period, validation_period):
        training_tickers = [Ticker(0, 'BTC', 'USDT'),
                            Ticker(0, 'ETH', 'USDT'),
                            Ticker(0, 'LTC', 'BTC'),
                            Ticker(0, 'ZEC', 'BTC'),
                            Ticker(0, 'ETC', 'BTC')]
        experiment_json = self.experiment_json_template.format(
            start_time=training_period.start_time_str, end_time=training_period.end_time_str)
        e = ExperimentManager(experiment_container=experiment_json, read_from_file=False)
        e.run_parallel_experiments()
        df = e.build_training_and_validation_dataframe(training_period, validation_period, training_tickers, 1,
                                                       "test.xlsx",
                                                       additional_fields={"grammar": "gv5"})
        return df



if __name__ == '__main__':
    training_period = Period('2018/04/01 00:00:00 UTC', '2018/05/01 00:00:00 UTC')
    validation_period = Period('2018/05/01 00:00:00 UTC', '2018/06/01 00:00:00 UTC')
    end_time = '2018/10/21 00:00:00 UTC'
    step = 60*60*24*7

    val = SlidingWindowValidator('gv5_experiments_sliding_template.json')
    val.run(training_period, validation_period, step, end_time, 'sliding_window_experiments_update_3.xlsx')
