class DataSplitter:
    def __init__(self, data: list):
        self.data = data

    def create_data_split(self, trn_ratio=0.7, dev_ratio=0.15):
        if trn_ratio + dev_ratio > 1.0:
            raise Exception("Sum of ratios cannot exceed 1.0.")
        elif trn_ratio + dev_ratio == 1.0:
            raise Exception("Some data must be used as test data.")

        num_instances = len(self.data)

        trn = self.data[:int(num_instances * trn_ratio)]
        dev = self.data[len(trn):int(num_instances * dev_ratio)]
        tst = self.data[len(trn) + len(dev):]

        return trn, dev, tst

    def construct_task_data(self, task):
        pass
