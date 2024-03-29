import os

model_name = os.environ['MODEL_NAME'] if 'MODEL_NAME' in os.environ else 'DKN'
# Currently included model
assert model_name in ['NRMS', 'NAML', 'LSTUR', 'DKN', 'HiFiArk', 'TANR', 'FIM']


class BaseConfig():
    """
    General configurations appiled to all models
    """
    num_batches = 60000  # Number of batches to train
    num_batches_show_loss = 100  # Number of batchs to show loss
    # Number of batchs to check metrics on validation dataset
    num_batches_validate = 800
    batch_size = 128 
    learning_rate = 0.001
    validation_proportion = 0.1
    num_workers = 0  # Number of workers for data loading
    num_clicked_news_a_user = 30  # Number of sampled click history for each user
    num_words_title = 20
    num_words_abstract = 50
    word_freq_threshold = 3
    entity_freq_threshold = 3
    entity_confidence_threshold = 0.5
    negative_sampling_ratio = 4  # K
    dropout_probability = 0.2
    # Modify the following by the output of `src/dataprocess.py`
    num_words = 1 + 44774
    num_categories = 1 + 295
    num_entities = 1 + 14697
    num_users = 1 + 711222
    word_embedding_dim = 300
    category_embedding_dim = 100
    # Modify the following only if you use another dataset
    entity_embedding_dim = 100
    # For additive attention
    query_vector_dim = 200




class DKNConfig(BaseConfig):
    dataset_attributes = {"news": ['title', 'title_entities'], "record": []}
    # For CNN
    num_filters = 50
    window_sizes = [2, 3, 4]
    # TODO: currently context is not available
    use_context = False
