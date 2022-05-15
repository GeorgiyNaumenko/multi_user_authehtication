from sklearn.decomposition import FastICA


class ICASeparator:

    def __init__(self, users_num):
        self.users_num = users_num
        self.ica = FastICA(n_components=users_num)

    def separate(self, audios):
        return self.ica.fit_transform(audios)
