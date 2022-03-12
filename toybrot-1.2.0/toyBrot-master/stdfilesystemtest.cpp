
#ifdef EXPERIMENTAL

    #include <experimental/filesystem>

    int main()
    {
        std::experimental::filesystem::create_directory("hadouken");
        return 0;
    }

#else
    #include <filesystem>

    int main()
    {
        std::filesystem::create_directory("hadouken");
        return 0;
    }
#endif
