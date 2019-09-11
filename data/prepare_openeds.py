import json
import os

import h5py
import imageio
import numpy as np
from joblib import Parallel, delayed


class OpenEDSPreparator:
    FOLDER_SEMANTIC_SEGMENTATION = "Semantic_Segmentation_Dataset"
    FOLDER_GENERATIVE = "Generative_Dataset"
    FOLDER_SEQUENTIAL = "Sequence_Dataset"

    def __init__(self, base_path, limit=-1, verbose=False, n_jobs=8, out_filename="all.h5"):
        self.base_path = base_path
        self.limit = limit - 1 if limit > 0 else np.inf
        self.verbose = verbose
        self.n_jobs = n_jobs

        self.path_out = os.path.join(self.base_path, out_filename)

    def parallel_load_and_preprocess(self, img_ids, path_images):
        if self.verbose:
            verbose = 3
        else:
            verbose = 0
        result = Parallel(n_jobs=self.n_jobs, verbose=verbose)(delayed(self.load_and_preprocess)(img_id, path_images) for img_id in img_ids)
        result_filtered = [r for r in result if r is not None]
        images, filenames = zip(*result_filtered)

        n_errors = len(result) - len(result_filtered)
        return images, filenames, n_errors

    def load_and_preprocess(self, filename, path):
        path_image = os.path.join(path, filename)
        try:
            img = imageio.imread(path_image)
        except ValueError:
            print(f"Could not read file from {path_image}")
            return None

        if len(img.shape) > 2:
            # import matplotlib.pyplot as ply
            # ply.imshow(img)
            # ply.show()
            # flat = img.reshape(-1, 3)
            # for i in np.random.choice(len(flat), size=10):  # For computationel efficiency only check 10 entries
            #     We want to make sure that all values in channel 3 are the same, so we can omit them.
                # e = flat[i]
                # assert np.min(e) == np.max(e)
            img = np.mean(img, axis=2)

        # img = ImagePreprocessor.normalize(img)
        # We don't need the jpg in the filenameZ
        return img, filename[:-4]

    def create_dataset_images(self, path, img_ids, group_user, ds_name):
        images, filenames, n_errors = self.parallel_load_and_preprocess(img_ids, path)
        images = np.array(images)
        filenames = np.array(filenames).astype('S13')
        # We use np.float16 in order to save disk space.
        group_user.create_dataset(ds_name, data=images, dtype=np.uint8, chunks=(1, *images.shape[1:]))
        group_user.create_dataset(ds_name+"_filenames", data=filenames, dtype='S13', chunks=True)
        print(f"Dataset '{ds_name}' with {len(images)} images created.")
        if n_errors > 0:
            print(f"{n_errors} skipped images when creating dataset")
        return group_user

    def create_dataset_labels(self, path, img_ids, group_user, ds_name):
        labels = map(lambda label_id: np.load(os.path.join(path, label_id[:-3] + "npy")),
                     img_ids)
        labels = np.array(list(labels))
        filenames = np.array(img_ids).astype('S13')
        # we only have integer values 0, 1, 2, 3 so we can use a uint8
        group_user.create_dataset(ds_name, data=labels, dtype=np.uint8, chunks=(1, *labels.shape[1:]))
        group_user.create_dataset(ds_name+"_filenames", data=filenames, dtype='S13', chunks=True)
        print(f"Dataset '{ds_name}' with {len(labels)} labels created.")
        return group_user

    def run(self):
        file_out = h5py.File(self.path_out, "w")
        print(f"Processing data from folder {self.base_path} and saving data as h5 to {self.path_out}")

        for subset in ["validation", "train"]:
            print(f"Processing '{subset}'...")

            g_subset = file_out.create_group(subset)

            path_user_ids = "/home/marcel/projects/data/openeds/OpenEDS_{}_userID_mapping_to_images.json".format(
                subset)
            with open(path_user_ids, 'r') as f:
                user_ids = json.load(f)

            n = min([len(user_ids) - 1, self.limit])
            for i, user in enumerate(user_ids):
                print(f"Processing user {i} / {n}")
                user_id = user["id"]
                g = g_subset.create_group(user_id)

                path_images_ss = os.path.join(self.base_path, self.FOLDER_SEMANTIC_SEGMENTATION, subset, "images")
                g = self.create_dataset_images(path_images_ss, user["semantic_segmenation_images"], g, ds_name="images_ss")

                path_labels_ss = os.path.join(self.base_path, self.FOLDER_SEMANTIC_SEGMENTATION, subset,
                                              "labels")
                g = self.create_dataset_labels(path_labels_ss, user["semantic_segmenation_images"], g, ds_name="labels_ss")

                path_images_gen = os.path.join(self.base_path, self.FOLDER_GENERATIVE, subset)
                g = self.create_dataset_images(path_images_gen, user["generative_images"], g, ds_name="images_gen")

                path_images_seq = os.path.join(self.base_path, self.FOLDER_SEQUENTIAL, subset)
                g = self.create_dataset_images(path_images_seq, user["sequence_images"], g, ds_name="images_seq")

                if i > self.limit:
                    break

        subset = "test"
        g_subset = file_out.create_group(subset)
        path_user_ids = os.path.join(self.base_path, f"OpenEDS_{subset}_userID_mapping_to_images.json")
        with open(path_user_ids, 'r') as f:
            user_ids = json.load(f)
        print(f"Processing '{subset}'...")

        n = min([len(user_ids) - 1, self.limit])
        for i, user in enumerate(user_ids):
            print(f"Processing user {i} / {n}")
            user_id = user["id"]
            g = g_subset.create_group(user_id)

            path_images_ss = os.path.join(self.base_path, self.FOLDER_SEMANTIC_SEGMENTATION, subset, "images")
            g = self.create_dataset_images(path_images_ss, user["semantic_segmenation_images"], g, ds_name="images_ss")

            path_labels_gen = os.path.join(self.base_path, self.FOLDER_GENERATIVE, subset,
                                          "labels")
            g = self.create_dataset_labels(path_labels_gen, user["generative_images"], g, ds_name="labels_gen")

            path_images_seq = os.path.join(self.base_path, self.FOLDER_SEQUENTIAL, subset)
            g = self.create_dataset_images(path_images_seq, user["sequence_images"], g, ds_name="images_seq")

            if i > self.limit:
                break

        file_out.close()


if __name__ == "__main__":
    base_path = "/home/marcel/projects/data/openeds"
    # out_filename = "small.h5"
    out_filename = "190910_all.h5"
    limit = -1
    preparator = OpenEDSPreparator(base_path=base_path, verbose=True, limit=limit, n_jobs=4, out_filename=out_filename)
    preparator.run()
