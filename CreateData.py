import Essentials
import numpy as np

RL040420_Data = ["RL040420", 334, 3079, 112, 112]
m, X_Grayscale, X_Color, Y = Essentials.getData(*RL040420_Data)

x_training, y_training = Essentials.splitDataTraining(X_Color, Y)
x_cv, y_cv = Essentials.splitDataCV(X_Color, Y)
x_test, y_test = Essentials.splitDataTesting(X_Color, Y)

np.savez_compressed('color_RL040420.npz', x_training=x_training,
                    y_training=y_training,
                    x_cv=x_cv,
                    y_cv=y_cv,
                    x_test=x_test,
                    y_test=y_test)

x_training, y_training = Essentials.splitDataTraining(X_Grayscale, Y)
x_cv, y_cv = Essentials.splitDataCV(X_Grayscale, Y)
x_test, y_test = Essentials.splitDataTesting(X_Grayscale, Y)

np.savez_compressed('grayscale_RL040420.npz', x_training=x_training,
                    y_training=y_training,
                    x_cv=x_cv,
                    y_cv=y_cv,
                    x_test=x_test,
                    y_test=y_test)