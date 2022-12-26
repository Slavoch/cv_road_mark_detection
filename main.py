import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from Configs.configs import IMG_SHAPE, CLASSES
from omegaconf import OmegaConf
import argparse
import csv
import pandas as pd
import CV_algs


class DataConvertor:
    def __init__(self, yaml):
        self.yaml = yaml

    def configurate(self):
        conf = OmegaConf.load(self.yaml)

        self.destination_SIZE = conf.target_size
        self.exceptions = conf.exceptions

        self.destination_rgb = conf.source_rgb
        self.destination_mask = conf.source_mask
        destination_to_save = conf.destination_to_save
        self.destination_to_save = destination_to_save

        self.destination_to_save_rgb = os.path.join(destination_to_save, "rgb")
        try:
            os.mkdir(self.destination_to_save_rgb)
        except OSError as error:
            print(error)
        self.destination_to_save_mask = os.path.join(destination_to_save, "mask")
        try:
            os.mkdir(self.destination_to_save_mask)
        except OSError as error:
            print(error)
        self.destination_to_save_result = os.path.join(destination_to_save, "result")
        try:
            os.mkdir(self.destination_to_save_result)
        except OSError as error:
            print(error)
        self.destination_to_save_info = os.path.join(destination_to_save, "line_info")
        try:
            os.mkdir(self.destination_to_save_info)
        except OSError as error:
            print(error)

    def reduce_classification(self, line_class):
        if line_class == CLASSES.EMPY:
            return 0
        if line_class == CLASSES.SOLID:
            return 2
        if line_class == CLASSES.SOLID_SOLID:
            return 2
        if line_class == CLASSES.SOLID_DASH:
            return 2
        if line_class == CLASSES.DASH_SOLID:
            return 2
        if line_class == CLASSES.DASH:
            return 3
        return None

    def main_loop(self):
        list_dir = os.listdir(self.destination_rgb)
        Y_SHAPE, X_SHAPE = IMG_SHAPE
        name_counter = 0
        destination_SIZE = self.destination_SIZE
        for i in list_dir:
            if i in self.exceptions:
                continue
            img_num = os.path.splitext(i)[0]
            print(img_num)

            img_real = cv2.imread(os.path.join(self.destination_rgb, f"{img_num}.png"))
            img_real = cv2.cvtColor(img_real, cv2.COLOR_BGR2RGB)
            img_real = cv2.resize(
                img_real, (X_SHAPE, Y_SHAPE), interpolation=cv2.INTER_CUBIC
            )

            mask = cv2.imread(os.path.join(self.destination_mask, f"{img_num}.png"))
            mask = cv2.resize(mask, (X_SHAPE, Y_SHAPE), interpolation=cv2.INTER_CUBIC)

            warped, (M, Minv) = CV_algs.preprocess_image(mask, visualise=False)

            binary_mark = (np.sum(warped, axis=2) == 441).astype("uint8")
            kernel = np.ones((2, 2), "uint8")
            binary_mark = cv2.dilate(binary_mark, kernel, iterations=1)
            binary_roadbed = (np.sum(warped, axis=2) == 320).astype("uint8")
            binary_roadbed = np.logical_or(binary_mark, binary_roadbed).astype("uint8")
            kernel = np.ones((10, 10), "uint8")
            binary_roadbed = cv2.dilate(binary_roadbed, kernel, iterations=1)
            binary_roadbed = CV_algs.smalldeleteArreas(
                binary_roadbed, diagnostics=False
            )

            simple_test_output = CV_algs.simple_test(binary_mark, binary_roadbed)

            success_mark, poly_param, (lc, rc) = simple_test_output[0]
            success_roadbed, roadbed_fit = simple_test_output[1]
            img_poly, out = simple_test_output[2]

            result = img_real

            Mat = np.zeros(destination_SIZE)
            counter = 1
            lanes_class_information = ""
            if success_roadbed:
                result = CV_algs.draw(
                    img_real, warped, Minv, roadbed_fit, lineColor=(255, 0, 0)
                )

                Mat = CV_algs.addLine(Mat, IMG_SHAPE, Minv, roadbed_fit[0], counter)
                lanes_class_information += f"{1} "
                counter += 1
                Mat = CV_algs.addLine(Mat, IMG_SHAPE, Minv, roadbed_fit[1], counter)
                lanes_class_information += f"{1} "
                counter += 1

            if success_mark:
                result2 = CV_algs.draw(
                    img_real,
                    warped,
                    Minv,
                    poly_param,
                    color=(220, 0, 110),
                    lineColor=(0, 0, 0),
                )
                result = cv2.addWeighted(result, 0.5, result2, 0.5, 0)
                result = cv2.putText(
                    result,
                    str(lc),
                    (10, 100),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (0, 255, 0),
                    thickness=2,
                )
                result = cv2.putText(
                    result,
                    str(rc),
                    (IMG_SHAPE[1] - 300, 100),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (0, 255, 0),
                    thickness=2,
                )

                Mat = CV_algs.addLine(Mat, IMG_SHAPE, Minv, poly_param[0], counter)
                lanes_class_information += f"{self.reduce_classification(lc)} "
                counter += 1
                Mat = CV_algs.addLine(Mat, IMG_SHAPE, Minv, poly_param[1], counter)
                lanes_class_information += f"{self.reduce_classification(rc)} "
                counter += 1
                if (
                    self.reduce_classification(lc)
                    > 3 | self.reduce_classification(rc)
                    > 3
                ):
                    raise Exception("your classification order is broken")
            while counter <= 6:
                lanes_class_information += f"{0} "
                counter += 1
            result = cv2.putText(
                result,
                f"frame:{i}",
                (10, 50),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 255, 0),
                thickness=2,
            )

            img_for_save = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            if success_mark:
                cv2.imwrite(
                    os.path.join(
                        self.destination_to_save_rgb,
                        f"{name_counter}".zfill(5) + ".png",
                    ),
                    cv2.resize(img_real, (destination_SIZE[1], destination_SIZE[0])),
                )
                cv2.imwrite(
                    os.path.join(
                        self.destination_to_save_mask,
                        f"{name_counter}".zfill(5) + ".png",
                    ),
                    Mat,
                )
                cv2.imwrite(
                    os.path.join(
                        self.destination_to_save_result,
                        f"{name_counter}".zfill(5) + ".png",
                    ),
                    img_for_save,
                )
                f = open(
                    os.path.join(
                        self.destination_to_save_info,
                        f"{name_counter}".zfill(5) + ".txt",
                    ),
                    "w",
                )
                f.write(lanes_class_information)
                f.close()
                name_counter += 1
            print("done")

    def make_csv(self):
        rgb_paths = []
        mask_paths = []
        lane_infos = []
        list_dir = os.listdir(self.destination_to_save_mask)
        for name in list_dir:
            i = os.path.splitext(name)[0]
            rgb_paths.append(f"rgb/{i}.png")
            mask_paths.append(f"mask/{i}.png")

            with open(
                os.path.join(self.destination_to_save_info, f"{i}.txt"), "r"
            ) as f:
                line = f.readline()
                lane_infos.append(line[:-1])
        cv_name = os.path.join(self.destination_to_save, "DataSet.csv")
        pd.DataFrame(
            {
                "path_to_input": rgb_paths,
                "path_to_mask": mask_paths,
                "lane_classes": lane_infos,
            }
        ).to_csv(cv_name, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get config source")
    parser.add_argument(
        "--conf", dest="yaml", required=True, help="source of a .yaml file"
    )
    args = parser.parse_args()
    print(f"Init: Creating vars from {args.yaml}; mkdir for resulting files")
    convertor = DataConvertor(args.yaml)
    convertor.configurate()
    print(f"Init: Done")
    print(f"Convertion: Start convertion")
    convertor.main_loop()
    print(f"Convertion: Done")
    print(f"csv: making")
    convertor.make_csv()
    print(f"csv: done")
    print("All processes are successfully done")
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from Configs.configs import IMG_SHAPE, CLASSES
from omegaconf import OmegaConf
import argparse
import csv
import pandas as pd
import CV_algs


class DataConvertor:
    def __init__(self, yaml):
        self.yaml = yaml

    def configurate(self):
        conf = OmegaConf.load(self.yaml)

        self.destination_SIZE = conf.target_size
        self.exceptions = conf.exceptions

        self.destination_rgb = conf.source_rgb
        self.destination_mask = conf.source_mask
        destination_to_save = conf.destination_to_save
        self.destination_to_save = destination_to_save

        self.destination_to_save_rgb = os.path.join(destination_to_save, "rgb")
        try:
            os.mkdir(self.destination_to_save_rgb)
        except OSError as error:
            print(error)
        self.destination_to_save_mask = os.path.join(destination_to_save, "mask")
        try:
            os.mkdir(self.destination_to_save_mask)
        except OSError as error:
            print(error)
        self.destination_to_save_result = os.path.join(destination_to_save, "result")
        try:
            os.mkdir(self.destination_to_save_result)
        except OSError as error:
            print(error)
        self.destination_to_save_info = os.path.join(destination_to_save, "line_info")
        try:
            os.mkdir(self.destination_to_save_info)
        except OSError as error:
            print(error)

    def reduce_classification(self, line_class):
        if line_class == CLASSES.EMPY:
            return 0
        if line_class == CLASSES.SOLID:
            return 2
        if line_class == CLASSES.SOLID_SOLID:
            return 2
        if line_class == CLASSES.SOLID_DASH:
            return 2
        if line_class == CLASSES.DASH_SOLID:
            return 2
        if line_class == CLASSES.DASH:
            return 3
        return None

    def main_loop(self):
        list_dir = os.listdir(self.destination_rgb)
        Y_SHAPE, X_SHAPE = IMG_SHAPE
        name_counter = 0
        destination_SIZE = self.destination_SIZE
        for i in list_dir:
            if i in self.exceptions:
                continue
            img_num = os.path.splitext(i)[0]
            print(img_num)

            img_real = cv2.imread(os.path.join(self.destination_rgb, f"{img_num}.png"))
            img_real = cv2.cvtColor(img_real, cv2.COLOR_BGR2RGB)
            img_real = cv2.resize(
                img_real, (X_SHAPE, Y_SHAPE), interpolation=cv2.INTER_CUBIC
            )

            mask = cv2.imread(os.path.join(self.destination_mask, f"{img_num}.png"))
            mask = cv2.resize(mask, (X_SHAPE, Y_SHAPE), interpolation=cv2.INTER_CUBIC)

            warped, (M, Minv) = CV_algs.preprocess_image(mask, visualise=False)

            binary_mark = (np.sum(warped, axis=2) == 441).astype("uint8")
            kernel = np.ones((2, 2), "uint8")
            binary_mark = cv2.dilate(binary_mark, kernel, iterations=1)
            binary_roadbed = (np.sum(warped, axis=2) == 320).astype("uint8")
            binary_roadbed = np.logical_or(binary_mark, binary_roadbed).astype("uint8")
            kernel = np.ones((10, 10), "uint8")
            binary_roadbed = cv2.dilate(binary_roadbed, kernel, iterations=1)
            binary_roadbed = CV_algs.smalldeleteArreas(
                binary_roadbed, diagnostics=False
            )

            simple_test_output = CV_algs.simple_test(binary_mark, binary_roadbed)

            success_mark, poly_param, (lc, rc) = simple_test_output[0]
            success_roadbed, roadbed_fit = simple_test_output[1]
            img_poly, out = simple_test_output[2]

            result = img_real

            Mat = np.zeros(destination_SIZE)
            counter = 1
            lanes_class_information = ""
            if success_roadbed:
                result = CV_algs.draw(
                    img_real, warped, Minv, roadbed_fit, lineColor=(255, 0, 0)
                )

                Mat = CV_algs.addLine(Mat, IMG_SHAPE, Minv, roadbed_fit[0], counter)
                lanes_class_information += f"{1} "
                counter += 1
                Mat = CV_algs.addLine(Mat, IMG_SHAPE, Minv, roadbed_fit[1], counter)
                lanes_class_information += f"{1} "
                counter += 1

            if success_mark:
                result2 = CV_algs.draw(
                    img_real,
                    warped,
                    Minv,
                    poly_param,
                    color=(220, 0, 110),
                    lineColor=(0, 0, 0),
                )
                result = cv2.addWeighted(result, 0.5, result2, 0.5, 0)
                result = cv2.putText(
                    result,
                    str(lc),
                    (10, 100),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (0, 255, 0),
                    thickness=2,
                )
                result = cv2.putText(
                    result,
                    str(rc),
                    (IMG_SHAPE[1] - 300, 100),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (0, 255, 0),
                    thickness=2,
                )

                Mat = CV_algs.addLine(Mat, IMG_SHAPE, Minv, poly_param[0], counter)
                lanes_class_information += f"{self.reduce_classification(lc)} "
                counter += 1
                Mat = CV_algs.addLine(Mat, IMG_SHAPE, Minv, poly_param[1], counter)
                lanes_class_information += f"{self.reduce_classification(rc)} "
                counter += 1
                if (
                    self.reduce_classification(lc)
                    > 3 | self.reduce_classification(rc)
                    > 3
                ):
                    raise Exception("your classification order is broken")
            while counter <= 6:
                lanes_class_information += f"{0} "
                counter += 1
            result = cv2.putText(
                result,
                f"frame:{i}",
                (10, 50),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 255, 0),
                thickness=2,
            )

            img_for_save = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            if success_mark:
                cv2.imwrite(
                    os.path.join(
                        self.destination_to_save_rgb,
                        f"{name_counter}".zfill(5) + ".png",
                    ),
                    cv2.resize(img_real, (destination_SIZE[1], destination_SIZE[0])),
                )
                cv2.imwrite(
                    os.path.join(
                        self.destination_to_save_mask,
                        f"{name_counter}".zfill(5) + ".png",
                    ),
                    Mat,
                )
                cv2.imwrite(
                    os.path.join(
                        self.destination_to_save_result,
                        f"{name_counter}".zfill(5) + ".png",
                    ),
                    img_for_save,
                )
                f = open(
                    os.path.join(
                        self.destination_to_save_info,
                        f"{name_counter}".zfill(5) + ".txt",
                    ),
                    "w",
                )
                f.write(lanes_class_information)
                f.close()
                name_counter += 1
            print("done")

    def make_csv(self):
        rgb_paths = []
        mask_paths = []
        lane_infos = []
        list_dir = os.listdir(self.destination_to_save_mask)
        for name in list_dir:
            i = os.path.splitext(name)[0]
            rgb_paths.append(f"rgb/{i}.png")
            mask_paths.append(f"mask/{i}.png")

            with open(
                os.path.join(self.destination_to_save_info, f"{i}.txt"), "r"
            ) as f:
                line = f.readline()
                lane_infos.append(line[:-1])
        cv_name = os.path.join(self.destination_to_save, "DataSet.csv")
        pd.DataFrame(
            {
                "path_to_input": rgb_paths,
                "path_to_mask": mask_paths,
                "lane_classes": lane_infos,
            }
        ).to_csv(cv_name, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get config source")
    parser.add_argument(
        "--conf", dest="yaml", required=True, help="source of a .yaml file"
    )
    args = parser.parse_args()
    print(f"Init: Creating vars from {args.yaml}; mkdir for resulting files")
    convertor = DataConvertor(args.yaml)
    convertor.configurate()
    print(f"Init: Done")
    print(f"Convertion: Start convertion")
    convertor.main_loop()
    print(f"Convertion: Done")
    print(f"csv: making")
    convertor.make_csv()
    print(f"csv: done")
    print("All processes are successfully done")
