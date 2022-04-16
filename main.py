import time

import cv2
import fingerprint_enhancer
import glob
import matplotlib.pyplot as plt

import fingerprint_feature_extractor
import numpy as np
from PIL import Image
import json


class FingerPrintAnalyzer:
    def __init__(self):
        self.fingerprints = []  # [path_to_fingerprint, label]
        self.fp_features = None

    def __pairs_to_binary(self, set):
        bins = np.histogram(set, bins=31)[1]        # 31 bins because we want to achieve a 5 bits representation
        quantized_values = np.digitize(set, bins) - 1

        binary_values = []
        for i in range(len(quantized_values)):
            bin = np.binary_repr(quantized_values[i], width=5)
            binary_values.append(bin)

        return binary_values

    def __get_feature_vector_fingerprint(self, path):
        sample = f"{path.split('/')[0]}/{path.split('/')[2].split('-')[0]}/{path.split('/')[2].split('-')[1]}"
        x_set, y_set, phi_set = self.fp_features[sample]
        if len(x_set) == 0:
            feature_vector = np.zeros(np.power(2, 15), dtype=int)
            return "".join(map(str, feature_vector)), 1

        l_set = []
        alpha_set = []
        beta_set = []

        for i in range(len(x_set) - 1):
            for j in range(i + 1, len(x_set)):
                x1 = x_set[i]
                x2 = x_set[j]

                y1 = y_set[i]
                y2 = y_set[j]

                phi1 = phi_set[i]
                phi2 = phi_set[j]

                x = (x2 - x1) * np.cos(phi1) - (y2 - y1) * np.sin(phi1)
                y = (x2 - x1) * np.sin(phi1) - (y2 - y1) * np.cos(phi1)

                l = np.sqrt(np.square(x) + np.square(y))
                if x == 0:
                    continue
                alpha1 = np.arctan(y / x)
                beta2 = alpha1 + phi2 - phi1

                l_set.append(l)
                alpha_set.append(alpha1)
                beta_set.append(beta2)

        l_set_bin = self.__pairs_to_binary(l_set)
        alpha_set_bin = self.__pairs_to_binary(alpha_set)
        beta_set_bin = self.__pairs_to_binary(beta_set)

        feature_vector = np.zeros(np.power(2, 15), dtype=int)
        for i in range(len(l_set_bin)):
            bin = int(l_set_bin[i] + alpha_set_bin[i] + beta_set_bin[i], 2)
            feature_vector[bin] = 1

        number_of_features = sum(feature_vector)

        return "".join(map(str, feature_vector)), number_of_features

    def apply_transformation(self):
        # TEMPORARY IMPLEMENTATION
        Original_Image = Image.open(f"data/DB1_B/101_1.tif")

        rotated_image1 = Original_Image.rotate(1)
        rotated_image1.save(f"data/DB1_B/101_r_1.tif", quality=100, subsampling=0)
        rotated_image1 = Original_Image.rotate(2)
        rotated_image1.save(f"data/DB1_B/101_r_2.tif", quality=100, subsampling=0)
        rotated_image1 = Original_Image.rotate(5)
        rotated_image1.save(f"data/DB1_B/101_r_3.tif", quality=100, subsampling=0)
        rotated_image1 = Original_Image.rotate(10)
        rotated_image1.save(f"data/DB1_B/101_r_4.tif", quality=100, subsampling=0)
        rotated_image1 = Original_Image.rotate(12)
        rotated_image1.save(f"data/DB1_B/101_r_5.tif", quality=100, subsampling=0)
        rotated_image1 = Original_Image.rotate(15)
        rotated_image1.save(f"data/DB1_B/101_r_6.tif", quality=100, subsampling=0)
        rotated_image1 = Original_Image.rotate(18)
        rotated_image1.save(f"data/DB1_B/101_r_7.tif", quality=100, subsampling=0)
        rotated_image1 = Original_Image.rotate(20)
        rotated_image1.save(f"data/DB1_B/101_r_8.tif", quality=100, subsampling=0)
        rotated_image1 = Original_Image.rotate(25)
        rotated_image1.save(f"data/DB1_B/101_r_9.tif", quality=100, subsampling=0)
        rotated_image1 = Original_Image.rotate(30)
        rotated_image1.save(f"data/DB1_B/101_r_10.tif", quality=100, subsampling=0)
        rotated_image1 = Original_Image.rotate(35)
        rotated_image1.save(f"data/DB1_B/101_r_11.tif", quality=100, subsampling=0)

    def __compare_pair_of_fingerprints(self, fp1, fp2):
        # img1 = cv2.imread(fp1, 0)
        # img2 = cv2.imread(fp2, 0)

        feature_vector_1, number_of_features_1 = self.__get_feature_vector_fingerprint(fp1)
        feature_vector_2, number_of_features_2 = self.__get_feature_vector_fingerprint(fp2)

        hamming_distance = sum(c1 != c2 for c1, c2 in zip(feature_vector_1, feature_vector_2))
        processed_hamming_distance = (number_of_features_1 + number_of_features_2 - hamming_distance) / (number_of_features_1 + number_of_features_2)

        return processed_hamming_distance

    def initialize_fingerprints_labels(self):
        items_db = glob.glob("data/enhanced_images/*")

        for fp in items_db:
            label = f"{fp.split('/')[2].split('_')[0]}_{fp.split('/')[2].split('_')[1]}"
            self.fingerprints.append([fp, label])
        self.fingerprints.sort()
        with open("data/features_dict.json", 'r') as f:
            self.fp_features = json.load(f)

    def __calculate_hamming_distances_all_pairs(self):
        hamming_distances_same_fp = []
        hamming_distances_different_fp = []
        counter = 0
        number_of_steps = 100
        for i in range(number_of_steps-1):
            for j in range(i+1, number_of_steps):
                counter += 1
                print(f"Step: {counter} of {int(number_of_steps*(number_of_steps-1)/2)} - Comparing {self.fingerprints[i][1]} and {self.fingerprints[j][1]}")
                comparison_bool = self.fingerprints[i][1] == self.fingerprints[j][1]
                hamming_distance = self.__compare_pair_of_fingerprints(self.fingerprints[i][0], self.fingerprints[j][0])
                if comparison_bool:
                    hamming_distances_same_fp.append(hamming_distance)
                else:
                    hamming_distances_different_fp.append(hamming_distance)
        return hamming_distances_same_fp, hamming_distances_different_fp

    def __calculate_hamming_distances(self):
        hamming_distances_same_fp = []
        hamming_distances_different_fp = []
        counter = 0
        number_of_steps = len(self.fingerprints)
        for i in range(1, number_of_steps):
            counter += 1
            print(f"Step: {counter} of {number_of_steps} - Comparing {self.fingerprints[0][1]} and {self.fingerprints[i][1]}")
            comparison_bool = self.fingerprints[0][1] == self.fingerprints[i][1]
            hamming_distance = self.__compare_pair_of_fingerprints(self.fingerprints[0][0], self.fingerprints[i][0])
            if comparison_bool:
                hamming_distances_same_fp.append(hamming_distance)
            else:
                hamming_distances_different_fp.append(hamming_distance)
        return hamming_distances_same_fp, hamming_distances_different_fp

    def __plot_histogram(self, hd_same_fp, hd_diff_fp):
        n, bins, patches = plt.hist(x=hd_same_fp, bins='auto', color='#0504aa',
                                    alpha=0.7, rwidth=0.85)
        n_1, bins_1, patches_1 = plt.hist(x=hd_diff_fp, bins='auto', color='#FF0000',
                                    alpha=0.7, rwidth=0.85)
        plt.xlim(0, 0.2)
        plt.show()

    def analyze_fingerprints(self):
        hamming_distances_same_fp, hamming_distances_different_fp = self.__calculate_hamming_distances()
        self.__plot_histogram(hamming_distances_same_fp, hamming_distances_different_fp)

    def __load_raw_images(self):
        items_db1_rotated = glob.glob("data/DB1_B_rotated/*")
        items_db1 = glob.glob("data/DB1_B/*")
        items_db2 = glob.glob("data/DB2_B/*")
        items_db3 = glob.glob("data/DB3_B/*")
        items_db4 = glob.glob("data/DB4_B/*")

        for fp in items_db1_rotated:
            label = f"{fp.split('/')[1]}-{fp.split('/')[2].split('_')[0]}"
            self.fingerprints.append([fp, label])
        for fp in items_db1:
            label = f"{fp.split('/')[1]}-{fp.split('/')[2].split('_')[0]}"
            self.fingerprints.append([fp, label])
        for fp in items_db2:
            label = f"{fp.split('/')[1]}-{fp.split('/')[2].split('_')[0]}"
            self.fingerprints.append([fp, label])
        for fp in items_db3:
            label = f"{fp.split('/')[1]}-{fp.split('/')[2].split('_')[0]}"
            self.fingerprints.append([fp, label])
        for fp in items_db4:
            label = f"{fp.split('/')[1]}-{fp.split('/')[2].split('_')[0]}"
            self.fingerprints.append([fp, label])

    def enhance_and_save_images(self):
        self.__load_raw_images()

        for i in range(len(self.fingerprints)):
            print(f"Step {i+1} of {len(self.fingerprints)}")
            img = cv2.imread(self.fingerprints[i][0], 0)  # read the input image --> You can enhance the fingerprint image using the "fingerprint_enhancer" library
            try:
                out = fingerprint_enhancer.enhance_Fingerprint(img)  # enhance the fingerprint image
            except IndexError:
                continue
            cv2.imwrite(f"data/enhanced_images/{self.fingerprints[i][0].split('/')[-2]}-{self.fingerprints[i][0].split('/')[-1]}", out)

    def precalculate_features(self):
        self.__load_raw_images()

        features = {}
        for i in range(len(self.fingerprints)):
            print(f"Step {i+1} of {len(self.fingerprints)}")
            img = cv2.imread(self.fingerprints[i][0], 0)  # read the input image --> You can enhance the fingerprint image using the "fingerprint_enhancer" library
            try:
                out = fingerprint_enhancer.enhance_Fingerprint(img)  # enhance the fingerprint image
            except IndexError:
                continue
            FeaturesTerminations, FeaturesBifurcations = fingerprint_feature_extractor.extract_minutiae_features(out, showResult=False)
            x_set = []
            y_set = []
            phi_set = []

            for j in range(len(FeaturesTerminations)):
                x, y, phi = FeaturesTerminations[j].locX, FeaturesTerminations[j].locY, np.radians(
                    FeaturesTerminations[j].Orientation[0])
                x_set.append(float(x))
                y_set.append(float(y))
                phi_set.append(float(phi))

            # for j in range(len(FeaturesBifurcations)):
            #     x1, y1, phi1 = FeaturesBifurcations[j].locX, FeaturesTerminations[j].locY, np.radians(FeaturesTerminations[j].Orientation[0])
            #     x_set.append(x1)
            #     y_set.append(y1)
            #     phi_set.append(phi1)

            features[self.fingerprints[i][0]] = [x_set, y_set, phi_set]
        json_dump = json.dumps(features)
        f = open("data/features_dict.json", "w")
        f.write(json_dump)
        f.close()


if __name__ == '__main__':
    fp = FingerPrintAnalyzer()

    # analyze the fingerprints against each other
    fp.initialize_fingerprints_labels()
    fp.analyze_fingerprints()

    # ONE TIME RUN
    # fp.apply_transformation()
    # fp.enhance_and_save_images()
    # fp.precalculate_features()
