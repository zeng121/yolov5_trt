#pragma once
#include<dirent.h>
#include<cstring>
#include<cassert>
#include<sys/stat.h>

static inline int read_files_in_dir(const char* p_dir_name, std::vector<std::string>& file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}

static bool ifFileExists(const char *FileName)
{
    struct stat my_stat;
    return (stat(FileName, &my_stat) == 0);
}

static size_t FileSize(const char* FileName)
{
    assert(ifFileExists(FileName));
    struct stat my_stat;
    stat(FileName,&my_stat);
    return my_stat.st_size;
}