from gp_artemis import ExperimentManager
from comparative_evaluation import Ticker
import pandas as pd
from dateutil import parser
import os
import pickle
import logging

from gp_utils import Period


class SlidingWindowValidator:

    def __init__(self, experiment_json_template_path):
        with open(experiment_json_template_path, 'r') as f:
            self.experiment_json_template = f.read()

    def run(self, training_period, validation_period, step, end_time_str, out_filename='sliding_window'):

        pickle_filename = out_filename + '_df.pkl' if out_filename is not None else None

        # in case we already have this dataframe saved, return that
        if pickle_filename is not None and os.path.exists(pickle_filename):
            logging.info(f'File {pickle_filename} already exists, returning saved dataframe...')
            return pd.read_pickle(pickle_filename)

        end_time = parser.parse(end_time_str).timestamp()
        dataframes = []

        while(validation_period.end_time < end_time):
            df = self._run_experiments_and_get_results(training_period, validation_period)
            training_period = training_period.get_shifted_forward(step)
            validation_period = validation_period.get_shifted_forward(step)
            dataframes.append(df)

        df = pd.concat(dataframes)
        if out_filename is not None:
            writer = pd.ExcelWriter(out_filename + '.xlsx')
            df.to_excel(writer, 'Results')
            writer.save()
            writer.close()
            df.to_pickle(pickle_filename)
        return df


    def _run_experiments_and_get_results(self, training_period, validation_period, top_n=5):
        training_tickers = [Ticker(0, 'BTC', 'USDT'),
                            Ticker(0, 'ETH', 'USDT'),
                            Ticker(0, 'LTC', 'BTC'),
                            Ticker(0, 'ZEC', 'BTC'),
                            Ticker(0, 'ETC', 'BTC')]
        experiment_json = self.experiment_json_template.format(
            start_time=training_period.start_time_str, end_time=training_period.end_time_str)
        e = ExperimentManager(experiment_container=experiment_json, read_from_file=False)
        e.run_parallel_experiments()
        df = e.build_training_and_validation_dataframe(training_period, validation_period, training_tickers, top_n,
                                                       additional_fields={"grammar": "gv5"})
        return df

    def recreate_individuals(self, df):
        training_tickers = [Ticker(0, 'BTC', 'USDT'),
                            Ticker(0, 'ETH', 'USDT'),
                            Ticker(0, 'LTC', 'BTC'),
                            Ticker(0, 'ZEC', 'BTC'),
                            Ticker(0, 'ETC', 'BTC')]

        row = df.iloc[0]
        training_start, training_end = row.training_period.split(' - ')
        validation_start, validation_end = row.validation_period.split(' - ')


        experiment_json = self.experiment_json_template.format(
            start_time=training_start, end_time=training_end)
        e = ExperimentManager(experiment_container=experiment_json, read_from_file=False)

        e.build_genetic_program(data=None, function_provider=e.function_provider, db_record=db_record)



if __name__ == '__main__':
    training_period = Period('2018/04/01 00:00:00 UTC', '2018/05/01 00:00:00 UTC')
    validation_period = Period('2018/05/01 00:00:00 UTC', '2018/06/01 00:00:00 UTC')
    end_time = '2018/10/21 00:00:00 UTC'
    step = 60*60*24*7

    val = SlidingWindowValidator('gv5_experiments_sliding_template.json')
    val.run(training_period, validation_period, step, end_time, 'sliding_window_experiments_update_5')
