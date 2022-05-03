class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, df,
                 batch_size=32,
                 input_size=(120, 120, 3),
                 shuffle=True):
        
        self.df = df.copy()
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.n = len(self.df)
        
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def __get_input(self, path, target_size):

        image = tf.keras.preprocessing.image.load_img(path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)

        image_arr = tf.image.resize(image_arr,(target_size[0], target_size[1])).numpy()

        return image_arr/255.
    
    def __get_output(self, label):
        return np.array([label])
    
    def __get_data(self, batches):
        # Generates data containing batch_size samples

        path_batch = batches['image']
                
        age_batch = batches['age']
        gender_batch = batches['gender']

        X_batch = np.asarray([self.__get_input(x, self.input_size) for x in path_batch])

        y0_batch = np.asarray([self.__get_output(y) for y in age_batch])
        y1_batch = np.asarray([self.__get_output(y) for y in gender_batch])

        return X_batch, tuple([y0_batch, y1_batch])
    
    def __getitem__(self, index):
        
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)        
        return X, y
    
    def __len__(self):
        return self.n // self.batch_size
    
    
traingen = CustomDataGen(train_df)

valgen = CustomDataGen(test_df,shuffle=False)
