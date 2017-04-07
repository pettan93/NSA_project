import os
import string
from subprocess import check_output

RESOURCES_DIR_PATH = '../resources/'


def generate_captcha(text, output_path, settings_dict=None):
    settings_args = ["height",  # default 60
                     "width",  # default (number of text characters * 60)
                     "fontSize",  # default 60
                     "rotationAmplitude",  # default 30
                     "scaleAmplitude",  # default 40
                     "showGrid",  # default False
                     "gridSize"  # default 10
                     ]

    if settings_dict is None:
        settings_dict = dict()
    for arg in settings_args:
        if arg not in settings_dict:
            settings_dict[arg] = "d"  # "d" stands for "default"

    check_output(
        "java -jar "
        + RESOURCES_DIR_PATH + "captcha_generator.jar "
        + text + " "
        + RESOURCES_DIR_PATH + "output/" + output_path + " "
        + RESOURCES_DIR_PATH + "fonts/boxpot.ttf "
        + settings_dict["height"] + " "
        + settings_dict["width"] + " "
        + settings_dict["fontSize"] + " "
        + settings_dict["rotationAmplitude"] + " "
        + settings_dict["scaleAmplitude"] + " "
        + settings_dict["showGrid"] + " "
        + settings_dict["gridSize"] + "",
        shell=True)


def traning_data_alphabet(samples_per_letter, settings):
    for index, char in enumerate(string.ascii_lowercase):
        print("Generating letter '" + char + "' [" + str(index + 1) + "/" + str(len(string.ascii_lowercase)) + "]")

        cap_char = char.capitalize()

        lowercase_dir = RESOURCES_DIR_PATH + "output/alphabet/lowercase/" + char
        os.makedirs(lowercase_dir)
        for i in range(samples_per_letter):
            print(" - lowercase image [" + str(i + 1) + "/" + str(samples_per_letter) + "]")
            generate_captcha(char, lowercase_dir + "/" + str(i + 1) + ".png", settings)

        uppercase_dir = RESOURCES_DIR_PATH + "output/alphabet/uppercase/" + cap_char
        os.makedirs(uppercase_dir)
        for i in range(samples_per_letter):
            print(" - uppercase image [" + str(i + 1) + "/" + str(samples_per_letter) + "]")
            generate_captcha(cap_char, uppercase_dir + "/" + str(i + 1) + ".png", settings)


# usages
# generate_captcha("foo", "out1.png", {"showGrid": "true", "fontSize": "200"})
# generate_captcha("bar", "out2.png")


# traning_data_alphabet(100, {"height": "100", "width": "100"})
generate_captcha("bard", "out4.png")
