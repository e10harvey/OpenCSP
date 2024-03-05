import copy
import cv2
import hashlib
import os
import numpy as np
from PIL import Image
import sys
import time

import FileCache as fc
import FileFingerprint as ff
import opencsp.common.lib.file.SimpleCsv as sc
from opencsp.common.lib.opencsp_path import opencsp_settings
import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
import opencsp.common.lib.process.subprocess_tools as st
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.hdf5_tools as h5
import opencsp.common.lib.tool.image_tools as it
import opencsp.common.lib.tool.log_tools as lt
import opencsp.common.lib.tool.time_date_tools as tdt
import SensitiveStringMatcher as ssm


class SensitiveStringsSearcher():
    _text_file_extensions = ['.txt', '.csv', '.py', '.md', '.rst']
    _text_file_names = ['.coverageac']

    def __init__(self, root_search_dir: str, sensitive_strings_csv: str, allowed_binary_files_csv: str, cache_file_csv: str = None):
        self.root_search_dir = root_search_dir
        self.sensitive_strings_csv = sensitive_strings_csv
        self.allowed_binary_files_csv = allowed_binary_files_csv
        self.cache_file_csv = cache_file_csv
        self.interactive = False
        self.date_time_str = tdt.current_date_time_string_forfile()
        self.tmp_dir_base = ft.norm_path(os.path.join(orp.opencsp_temporary_dir(), "SensitiveStringSearcher"))
        self.git_files_only = True

        self.matchers = self.build_matchers()
        self.matches: dict[str, list[ssm.Match]] = {}
        self.allowed_binary_files: list[ff.FileFingerprint] = []
        self.accepted_binary_files: list[ff.FileFingerprint] = []
        self.unknown_binary_files: list[ff.FileFingerprint] = []
        self.unfound_allowed_binary_files: list[ff.FileFingerprint] = []
        self.cached_cleared_files: list[fc.FileCache] = []
        self.new_cached_cleared_files: list[fc.FileCache] = []

    def __del__(self):
        if ft.directory_exists(self.tmp_dir_base):
            tmp_dirs = ft.files_in_directory(self.tmp_dir_base, files_only=False)
            tmp_dirs = list(filter(lambda s: s.startswith("tmp_"), tmp_dirs))
            tmp_dirs = list(filter(lambda s: os.path.isdir(s), tmp_dirs))
            for tmp_dir in tmp_dirs:
                ft.delete_files_in_directory(tmp_dir, "*")
                os.rmdir(tmp_dir)

    def build_matchers(self):
        matchers: list[ssm.SensitiveStringMatcher] = []

        path, name, ext = ft.path_components(self.sensitive_strings_csv)
        csv = sc.SimpleCsv("Sensitive Strings", path, name + ext)
        for row in csv.rows:
            name = list(row.values())[0]
            patterns = list(row.values())[1:]
            matchers.append(ssm.SensitiveStringMatcher(name, *patterns))

        return matchers

    def norm_path(self, file_path, file_name_ext: str):
        return ft.norm_path(os.path.join(self.root_search_dir, file_path, file_name_ext))

    def parse_file(self, file_path: str, file_name_ext: str):
        file_path_norm: str = self.norm_path(file_path, file_name_ext)
        lt.debug(file_path_norm)
        errmsg = ""
        is_binary_file = False

        # check if a binary file
        path, name, ext = ft.path_components(file_name_ext)
        ext = ext.lower()
        if self._is_img_ext(ext):
            if ext in self._text_file_extensions:
                is_binary_file = False
            elif f"{file_path}/{file_name_ext}" in self._text_file_names:
                is_binary_file = False
            else:
                is_binary_file = True
        else:
            # attempt to parse the file as a text file
            try:
                lines = ft.read_text_file(file_path_norm)
            except UnicodeDecodeError:
                errmsg = f"    UnicodeDecodeError in sensitive_strings.search_file: assuming is a binary file \"{file_path_norm}\""
                is_binary_file = True

        # register binary files
        if is_binary_file:
            file_ff = ff.FileFingerprint.for_file(self.root_search_dir, file_path, file_name_ext)

            if file_ff in self.allowed_binary_files:
                self.unfound_allowed_binary_files.remove(file_ff)
                self.accepted_binary_files.append(file_ff)
            else:
                if errmsg != "":
                    lt.warn(errmsg)
                # we'll deal with unknown files as a group
                self.unknown_binary_files.append(file_ff)

            # don't return anything
            lines = []

        return lines

    def search_lines(self, lines: list[str]):
        matches: list[ssm.Match] = []

        for matcher in self.matchers:
            matches += matcher.check_lines(lines)

        return matches

    def search_file(self, file_path: str, file_name_ext: str):
        lines = self.parse_file(file_path, file_name_ext)

        matches: list[ssm.Match] = []
        matches += self.search_lines([file_path + "/" + file_name_ext])
        matches += self.search_lines(lines)
        return matches

    def get_tmp_dir(self):
        i = 0
        while True:
            ret = ft.norm_path(os.path.join(self.tmp_dir_base, f"tmp_{i}"))
            if ft.directory_exists(ret):
                i += 1
            else:
                return ret

    def search_hdf5_file(self, hdf5_file: ff.FileFingerprint):
        norm_path = self.norm_path(hdf5_file.relative_path, hdf5_file.name_ext)
        relative_path_name_ext = f"{hdf5_file.relative_path}/{hdf5_file.name_ext}"
        matches: list[ssm.Match] = []

        # Extract the contents from the HDF5 file
        unzip_dir = self.get_tmp_dir()
        lt.info("")
        lt.info(f"**Extracting HDF5 file to {unzip_dir}**")
        h5_dir = h5.unzip(norm_path, unzip_dir)

        # Create a temporary allowed binary strings file
        fd, tmp_allowed_binary_csv = ft.get_temporary_file(".csv")
        with open(fd, "w") as fout:
            with open(self.allowed_binary_files_csv, "r") as fin:
                fout.write(fin.readline())

        # Create a searcher for the unzipped directory
        hdf5_searcher = SensitiveStringsSearcher(h5_dir, self.sensitive_strings_csv, tmp_allowed_binary_csv)
        hdf5_searcher.interactive = self.interactive
        hdf5_searcher.date_time_str = self.date_time_str
        hdf5_searcher.tmp_dir_base = self.tmp_dir_base
        hdf5_searcher.git_files_only = False

        # Validate all of the unzipped files
        error = hdf5_searcher.search_files()
        hdf5_matches = hdf5_searcher.matches
        if error != 0:
            # There was an error, but the user may want to sign off on the file anyways.
            if len(hdf5_matches) > 0:
                # Describe the issues with the HDF5 file
                lt.warn(f"Found {len(hdf5_matches)} possible issues with the HDF5 file '{relative_path_name_ext}':")
                prev_relpath_name_ext = None
                for file_relpath_name_ext in hdf5_matches:
                    if prev_relpath_name_ext != file_relpath_name_ext:
                        lt.warn(f"    {file_relpath_name_ext}:")
                        prev_relpath_name_ext = file_relpath_name_ext
                    for match in hdf5_matches[file_relpath_name_ext]:
                        lt.warn(f"        {match.msg} (line {match.lineno}, col {match.colno})")

                # Ask the user about signing off
                if self.interactive:
                    lt.info("Do you want to sign off on this file anyways (y/n)?")
                    val = input("")[0]
                    lt.info(f"    User responded '{val}'")
                    if val.lower() == 'y':
                        matches = []
                    else:
                        matches.append(ssm.Match(0, 0, 0, "", "", None, "HDF5 file denied by user"))
                else:  # if self.interactive
                    for file_relpath_name_ext in hdf5_matches:
                        match = hdf5_matches[file_relpath_name_ext]
                        path, name, _ = ft.path_components(file_relpath_name_ext)
                        dataset_name = path.replace("\\", "/") + "/" + name
                        match.msg = dataset_name + "::" + match.msg
                        matches.append(match)
            else:  # if len(hdf5_matches) > 0:
                lt.error_and_raise(RuntimeError, "Programmer error in SensitiveStringsSearcher.search_hdf5_files(): " +
                                   f"Errors were returned for file {relative_path_name_ext} but there were 0 matches found.")

        else:
            # There were no errors, matches should be empty
            if len(hdf5_matches) > 0:
                lt.error_and_raise(RuntimeError, "Programmer error in SensitiveStringsSearcher.search_hdf5_files(): " +
                                   f"No errors were returned for file {relative_path_name_ext} but there were {len(hdf5_matches)} > 0 matches found.")

        # Remove the temporary files created for the searcher.
        # Files created by the searcher should be removed in its __del__() method.
        ft.delete_file(tmp_allowed_binary_csv)

        return matches

    def search_binary_file(self, binary_file: ff.FileFingerprint) -> list[ssm.Match]:
        norm_path = self.norm_path(binary_file.relative_path, binary_file.name_ext)
        _, _, ext = ft.path_components(norm_path)
        relative_path_name_ext = f"{binary_file.relative_path}/{binary_file.name_ext}"
        matches: list[ssm.Match] = []

        if ext.lower().lstrip(".") in it.pil_image_formats_rw:
            if self.interactive:
                if self.interactive_image_sign_off(file_ff=binary_file):
                    return []
                else:
                    matches.append(ssm.Match(0, 0, 0, "", "", None, "File denied by user"))
            else:
                matches.append(ssm.Match(0, 0, 0, "", "", None, "Unknown image file"))

        elif ext.lower() == ".h5":
            matches += self.search_hdf5_file(binary_file)

        else:
            lt.info("")
            lt.info("Unknown binary file:")
            lt.info("    " + relative_path_name_ext)
            lt.info("Is this unknown binary file safe to add, and doesn't contain any sensitive information (y/n)?")
            val = input("")[0]
            lt.info(f"    User responded '{val}'")
            if val.lower() != 'y':
                matches.append(ssm.Match(0, 0, 0, "", "", None, "Unknown binary file"))

        return matches

    def _is_img_ext(self, ext: str):
        return ext.lower().lstrip(".") in it.pil_image_formats_rw

    def interactive_image_sign_off(self, np_image: np.ndarray = None, description: str = None, file_ff: ff.FileFingerprint = None) -> bool:
        if (np_image is None) and (file_ff is not None):
            file_norm_path = self.norm_path(file_ff.relative_path, file_ff.name_ext)
            _, name, ext = ft.path_components(file_norm_path)
            if self._is_img_ext(ext):
                img = Image.open(file_norm_path).convert('RGB')
                np_image = np.copy(np.array(img))
                img.close()
                return self.interactive_image_sign_off(np_image=np_image, description=f"{file_ff.relative_path}/{file_ff.name_ext}")
            else:
                return False

        else:
            # rescale the image for easier viewing
            img = it.numpy_to_image(np_image)
            rescaled = ""
            if img.size[0] > 1920:
                scale = 1920 / img.size[0]
                img = img.resize((int(scale * img.size[0]), int(scale * img.size[1])))
                np_image = np.array(img)
                rescaled = " (downscaled)"
            if img.size[0] > 1080:
                scale = 1080 / img.size[1]
                img = img.resize((int(scale * img.size[0]), int(scale * img.size[1])))
                np_image = np.array(img)
                rescaled = " (downscaled)"

            # Show the image and prompt the user
            lt.info("")
            lt.info("Is this image safe to add, and doesn't contain any sensitive information (y/n)?")
            cv2.imshow(description + rescaled, np_image)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            time.sleep(0.1)  # small delay to prevent accidental double-bounces

            # Check for 'y' or 'n'
            if key == ord('y') or key == ord('Y'):
                val = 'y'
            elif key == ord('n') or key == ord('N'):
                val = 'n'
            else:
                val = '?'
            if val.lower() in ["y", "n"]:
                lt.info(f"    User responded '{val}'")
            else:
                lt.error("Did not respond with either 'y' or 'n'. Assuming 'n'.")
                val = 'n'

            ret = val == 'y'
            return ret

    def _init_files_lists(self):
        self.matches.clear()

        abfc_p, abfc_n, abfc_e = ft.path_components(self.allowed_binary_files_csv)
        self.allowed_binary_files = [inst for inst, _ in ff.FileFingerprint.from_csv("Allowed Binary Files", abfc_p, abfc_n + abfc_e)]
        self.accepted_binary_files.clear()
        self.unknown_binary_files.clear()
        self.unfound_allowed_binary_files = copy.copy(self.allowed_binary_files)

        self.cached_cleared_files.clear()
        self.new_cached_cleared_files.clear()
        ss_p, ss_n, ss_e = ft.path_components(self.sensitive_strings_csv)
        sensitive_strings_cache = fc.FileCache.for_file(ss_p, "", ss_n + ss_e)
        if self.cache_file_csv != None and ft.file_exists(self.cache_file_csv):
            cp, cn, ce = ft.path_components(self.cache_file_csv)
            self.cached_cleared_files = [inst for inst, _ in fc.FileCache.from_csv("Cleared Files Cache", cp, cn + ce)]
            if not sensitive_strings_cache in self.cached_cleared_files:
                self.cached_cleared_files.clear()
        self.new_cached_cleared_files.append(sensitive_strings_cache)

    def search_files(self):
        self._init_files_lists()
        if self.git_files_only:
            git_stdout = st.run(
                "git ls-tree --full-tree --name-only -r HEAD", cwd=self.root_search_dir, stdout="collect", stderr="print")
            files = [line.val for line in git_stdout]
            lt.info(f"Searching for sensitive strings in {len(files)} tracked files")
        else:
            files = ft.files_in_directory(self.root_search_dir, files_only=True, recursive=True)
            lt.info(f"Searching for sensitive strings in {len(files)} files")
        files = sorted(files)

        # Search for sensitive strings in files
        matches: dict[str, list[ssm.Match]] = {}
        for file_path_name_ext in files:
            path, name, ext = ft.path_components(file_path_name_ext)
            file_cache = fc.FileCache.for_file(self.root_search_dir, path, name + ext)
            if not file_cache in self.cached_cleared_files:
                file_matches = self.search_file(path, name + ext)
                if len(file_matches) > 0:
                    matches[file_path_name_ext] = file_matches
                else:
                    self.cached_cleared_files.append(file_cache)
                    self.new_cached_cleared_files.append(file_cache)

        if len(matches) > 0:
            lt.error(f"Found {len(matches)} files containing sensitive strings:")
            for file in matches:
                lt.error(f"    File {file}:")
                for match in matches[file]:
                    lt.error(f"        {match.msg}")

        if len(self.unfound_allowed_binary_files) > 0:
            lt.error(f"Expected {len(self.unfound_allowed_binary_files)} binary files that aren't part of the git repository:")
            for file_ff in self.unfound_allowed_binary_files:
                lt.info("")
                lt.error(os.path.join(file_ff.relative_path, file_ff.name_ext))

        if len(self.unknown_binary_files) > 0:
            lt.warn(f"Found {len(self.unknown_binary_files)} unknown binary files that aren't part of the git repository:")
            unknowns_copy = copy.copy(self.unknown_binary_files)
            for file_ff in unknowns_copy:
                lt.info("")
                lt.info(os.path.join(file_ff.relative_path, file_ff.name_ext))
                num_signed_binary_files = 0
                parsable_matches: list[ssm.Match] = self.search_binary_file(file_ff)

                if len(parsable_matches) == 0:
                    # This file is ok.
                    # Add the validated and/or signed off file to the allowed binary files csv
                    self.unknown_binary_files.remove(file_ff)
                    self.allowed_binary_files.append(file_ff)

                    # First, make a backup copy of the allowed list csv file
                    if num_signed_binary_files == 0:
                        path, name, ext = ft.path_components(self.allowed_binary_files_csv)
                        ft.copy_file(self.allowed_binary_files_csv, path, f"{name}_backup_{self.date_time_str}{ext}")

                    # Overwrite the allowed list csv file with the updated allowed_binary_files
                    self.allowed_binary_files = sorted(self.allowed_binary_files)
                    file_ff.to_csv("Allowed Binary Files", path, name, rows=self.allowed_binary_files)

                    num_signed_binary_files += 1

                else:
                    # This file is not ok. Tell the user why.
                    lt.error(f"    Found {len(parsable_matches)} possible sensitive issues in file {self.norm_path(file_ff.relative_path, file_ff.name_ext)}.")
                    for _match in parsable_matches:
                        match: ssm.Match = _match
                        lt.error("    " + match.msg + f" (line {match.lineno}, col {match.colno})")

                # Date+time stamp the new allowed list csv files
                if num_signed_binary_files > 0:
                    path, name, ext = ft.path_components(self.allowed_binary_files_csv)
                    ft.copy_file(self.allowed_binary_files_csv, path, f"{name}_{self.date_time_str}{ext}")

        # Cache cleared files
        for file_ff in self.unknown_binary_files:
            for file_cf in self.new_cached_cleared_files:
                if file_ff.eq_aff(file_cf):
                    self.new_cached_cleared_files.remove(file_cf)
                    break
        if self.cache_file_csv != None and len(self.new_cached_cleared_files) > 0:
            path, name, ext = ft.path_components(self.cache_file_csv)
            ft.create_directories_if_necessary(path)
            self.new_cached_cleared_files[0].to_csv("Cleared Files Cache", path, name, rows=self.new_cached_cleared_files)

        # Executive summary
        info_or_warn = lt.info
        ret = len(matches) + len(self.unfound_allowed_binary_files) + len(self.unknown_binary_files)
        if ret > 0:
            info_or_warn = lt.warn
        info_or_warn("Summary:")
        info_or_warn("<<<PASS>>>" if ret == 0 else "<<<FAIL>>>")
        info_or_warn(f"Found {len(matches)} sensitive string matches")
        info_or_warn(f"Found {len(self.unknown_binary_files)} unknown binary files")
        info_or_warn(f"Did not find {len(self.unfound_allowed_binary_files)} expected binary files")

        # Add a 'match' for any unfound or unknown binary files
        for file_ff in self.unfound_allowed_binary_files:
            fpne = f"{file_ff.relative_path}/{file_ff.name_ext}"
            matches[fpne] = [] if (fpne not in matches) else matches[fpne]
            matches[fpne].append(ssm.Match(0, 0, 0, "", "", None, f"Unfound binary file {fpne}"))
        for file_ff in self.unknown_binary_files:
            fpne = f"{file_ff.relative_path}/{file_ff.name_ext}"
            matches[fpne] = [] if (fpne not in matches) else matches[fpne]
            matches[fpne].append(ssm.Match(0, 0, 0, "", "", None, f"Unknown binary file {fpne}"))

        self.matches = matches
        return ret


if __name__ == "__main__":
    ss_log_dir = ft.norm_path(opencsp_settings['sensitive_strings']['sensitive_strings_dir'])
    log_path = ft.norm_path(os.path.join(ss_log_dir, "sensitive_strings_log.txt"))
    sensitive_strings_csv = ft.norm_path(opencsp_settings['sensitive_strings']['sensitive_strings_file'])
    allowed_binary_files_csv = ft.norm_path(opencsp_settings['sensitive_strings']['allowed_binaries_file'])
    ss_cache_file = ft.norm_path(opencsp_settings['sensitive_strings']['cache_file'])
    date_time_str = tdt.current_date_time_string_forfile()

    log_already_exists = os.path.exists(log_path)
    path, name, ext = ft.path_components(log_path)
    log_path = os.path.join(path, f"{name}_{date_time_str}{ext}")
    lt.logger(log_path)

    root_search_dir = os.path.join(orp.opencsp_code_dir(), "..")
    searcher = SensitiveStringsSearcher(root_search_dir, sensitive_strings_csv, allowed_binary_files_csv, ss_cache_file)
    searcher.interactive = True
    searcher.date_time_str = date_time_str
    num_errors = searcher.search_files()

    if num_errors > 0:
        sys.exit(1)
    else:
        sys.exit(0)
