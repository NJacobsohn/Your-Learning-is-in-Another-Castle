import retro
import argparse

def play_movie(movie_path):
    movie = retro.Movie(movie_path)
    movie.step()

    env = retro.make(
        game=movie.get_game(),
        state=None,
        # bk2s can contain any button presses, so allow everything
        use_restricted_actions=retro.Actions.ALL,
        players=movie.players,
    )
    env.initial_state = movie.get_state()
    env.reset()

    while movie.step():
        keys = []
        for p in range(movie.players):
            for i in range(env.num_buttons):
                keys.append(movie.get_key(i, p))
        env.step(keys)
        env.render()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', "--movie", help="Filepath for a movie to watch")
    args = parser.parse_args()

    play_movie(args.movie)