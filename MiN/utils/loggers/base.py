class BaseExperimentLogger:
    def log_scalar(self, tag, value, step):
        return None

    def log_scalars(self, main_tag, tag_scalar_dict, step):
        return None

    def log_histogram(self, tag, values, step):
        return None

    def log_text(self, tag, text, step=0):
        return None

    def close(self):
        return None
