self_play_file_batch_size = 1 # e.g each batch has 10 files which is 10x100 games
self_play_file_increment = 1  #
training_repetition = 500  # every x training loops, save the checkpoint and evaluate the new model
sleep_seconds = 120 # when there is no self play files, sleep x seconds
mini_batch_size = 500
number_eval_games = 100


network_settings = {
    5 : {
        'cnnoutput' : 32,
        'num_blocks' : 2
    },

    9: {
        'cnnoutput': 128,
        'num_blocks': 2
    },

    13: {
        'cnnoutput': 128,
        'num_blocks': 3
    },

    19: {
        'cnnoutput': 32,
        'num_blocks': 4
    }
}
