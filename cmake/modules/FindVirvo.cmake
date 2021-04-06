include(FindPackageHandleStandardArgs)

set(paths
    /usr
    /usr/local
)

find_path(VIRVO_INCLUDE_DIR
    NAMES
        vvvirvo.h
    PATHS
        ${paths}
    PATH_SUFFIXES
        include
)

find_library(VIRVO_LIBRARY
    NAMES
        virvo
    PATHS
        ${paths}
    PATH_SUFFIXES
        lib64
        lib
)

find_library(VIRVO_FILEIO_LIBRARY
    NAMES
        virvo_fileio
    PATHS
        ${paths}
    PATH_SUFFIXES
        lib64
        lib
)


find_library(VIRVO_TRANSFUNC_LIBRARY
    NAMES
        virvo_transfunc
    PATHS
        ${paths}
    PATH_SUFFIXES
        lib64
        lib
)

set(VIRVO_LIBRARIES ${VIRVO_LIBRARY} ${VIRVO_FILEIO_LIBRARY} ${VIRVO_TRANSFUNC_LIBRARY})

find_package_handle_standard_args(Virvo
    VIRVO_DEFAULT_MSG
    VIRVO_INCLUDE_DIR
    VIRVO_LIBRARY
)
