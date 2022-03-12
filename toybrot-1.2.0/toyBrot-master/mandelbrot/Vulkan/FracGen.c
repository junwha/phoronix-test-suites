#include "FracGen.hpp"

#include <iostream>
#include <fstream>
#include <cfloat>
#include <string>

#include <vulkan/vulkan.h>


// The Vulkan instance and debug callback function
static VkInstance vkInst;
static VkDebugReportCallbackEXT debugCallback;
// The Physical device represents the actual hardware
// The vkDevice is essentially the wrapper which gives us access to it
// We also want to know about the memory we have available
static VkDevice vkDev;
static VkPhysicalDevice vkPhys;
static VkPhysicalDeviceMemoryProperties vkMProps;
// We need to specify a pipeline we can send command through
// In this case, it's a purely computational pipeline
static VkPipeline vkPpl;
static VkPipelineLayout vkPpLayout;
static VkShaderModule vkShMod;
// We submit commands throug a queue
// Commands get recorderd in a buffer and allocated in a pool
static VkCommandPool vkCmdPool;
static VkCommandBuffer vkCmdBuffer;
static VkQueue vkQ;
static uint32_t vkQIdx;
// Descriptors represent resources for shaders, a bit like the binding in OpenCL
// We organize these into sets
static VkDescriptorPool      vkDescPool;
static VkDescriptorSet       vkDescSet;
static VkDescriptorSetLayout vkDescLayout;

constexpr const float WORKGROUP_SIZE = 32;

static VKAPI_PTR VKAPI_CALL VkBool32 debugCallbackFn(VkDebugReportFlagsEXT      flags,
                                                     VkDebugReportObjectTypeEXT objectType,
                                                     uint64_t                   object,
                                                     size_t                     location,
                                                     int32_t                    messageCode,
                                                     const char*                pLayerPrefix,
                                                     const char*                pMessage,
                                                     void*                      pUserData);


struct vkPixFormat
{
    vkPixFormat(SDL_PixelFormat fmt)
        : Amask {fmt.Amask}
        , Rloss {fmt.Rloss}
        , Gloss {fmt.Gloss}
        , Bloss {fmt.Bloss}
        , Aloss {fmt.Aloss}
        , Rshift{fmt.Rshift}
        , Gshift{fmt.Gshift}
        , Bshift{fmt.Bshift}
        , Ashift{fmt.Ashift}
    {}

    uint32_t Amask,  Rloss,  Gloss
           , Bloss,  Aloss,  Rshift
           , Gshift, Bshift, Ashift;
};

uint32_t* readFile(uint32_t& length, const char* filename) {

        FILE* fp = fopen(filename, "rb");
        if (fp == nullptr) {
            printf("Could not find or open file: %s\n", filename);
        }

        // get file size.
        fseek(fp, 0, SEEK_END);
        long filesize = ftell(fp);
        fseek(fp, 0, SEEK_SET);

        long filesizepadded = long(ceil(filesize / 4.0)) * 4;

        // read file contents.
        char *str = new char[filesizepadded];
        fread(str, filesize, sizeof(char), fp);
        fclose(fp);

        // data padding.
        for (int i = filesize; i < filesizepadded; i++) {
            str[i] = 0;
        }

        length = filesizepadded;
        return (uint32_t *)str;
    }

#ifdef NDEBUG
    static constexpr const bool enableValidation = false;
#else
    static constexpr const bool enableValidation = true;
#endif

void FracGen::Generate(uint32_t* v, SDL_PixelFormat* format, int width, int height, Region r)
{
    /**
      * I'm leaving "all of this" here as a way of maintaining as much
      * "parity" with the other versions as possible. However, with Vulkan it is
      * particularly obvious that:
      * 1 - This would REALLY need to be broken up in digestible chunks
      * 2 - A lot of what happens in this call should be part of constructor/destructor instead
      */
    if(format == nullptr)
    {
        return;
    }

    const VkDeviceSize texSize = sizeof(uint32_t) * static_cast<size_t>(width) * static_cast<size_t>(height);
    const VkDeviceSize regSize = sizeof(double) * 4;
    const VkDeviceSize fmtSize = sizeof(vkPixFormat);



    /**
     * Create the buffer object to hold the data
     * and allocate the memory on the GPU
     */

    VkBuffer outBuf;
    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = texSize;
    bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if(vkCreateBuffer(vkDev, &bufInfo, nullptr, &outBuf) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create device buffer for output");
    }

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(vkDev, outBuf, &memReqs);

    //Acquire and allocate the memory for the buffer
    VkMemoryAllocateInfo memInfo{};
    memInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memInfo.allocationSize = memReqs.size;

    bool foundMem = false;
    for( uint32_t k = 0; k < vkMProps.memoryTypeCount; k++)
    {
        const VkMemoryType memoryType = vkMProps.memoryTypes[k];

        if (  (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT & memoryType.propertyFlags)
           && (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT & memoryType.propertyFlags)
           && (memReqs.size < vkMProps.memoryHeaps[memoryType.heapIndex].size)      )
        {
            memInfo.memoryTypeIndex = k;
            foundMem = true;
            break;
        }
    }

    if(!foundMem)
    {
        throw std::runtime_error("Failed to find adequate memory for output on the device");
    }

    VkDeviceMemory devVec;
    if(vkAllocateMemory(vkDev, &memInfo, nullptr, &devVec) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to allocate memory on devicefor output");
    }

    if(vkBindBufferMemory(vkDev, outBuf, devVec, 0) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to bind buffer memory for output");
    }


    /**
      * Same process, but now for the region
      */

    VkBuffer regBuf;
    VkBufferCreateInfo regInfo{};
    regInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    regInfo.size = regSize;
    regInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    regInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if(vkCreateBuffer(vkDev, &regInfo, nullptr, &regBuf) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create device buffer for Region");
    }

    vkGetBufferMemoryRequirements(vkDev, regBuf, &memReqs);

    //Acquire and allocate the memory for the buffer
    memInfo.allocationSize = memReqs.size;

    foundMem = false;
    for( uint32_t k = 0; k < vkMProps.memoryTypeCount; k++)
    {
        const VkMemoryType memoryType = vkMProps.memoryTypes[k];

        if (  (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT & memoryType.propertyFlags)
           && (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT & memoryType.propertyFlags)
           && (memReqs.size < vkMProps.memoryHeaps[memoryType.heapIndex].size)      )
        {
            memInfo.memoryTypeIndex = k;
            foundMem = true;
            break;
        }
    }

    if(!foundMem)
    {
        throw std::runtime_error("Failed to find adequate memory for Region on the device");
    }

    VkDeviceMemory devReg;
    if(vkAllocateMemory(vkDev, &regMemInfo, nullptr, &devReg) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to allocate memory on device for Region");
    }

    if(vkBindBufferMemory(vkDev, regBuf, devReg, 0) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to bind buffer memory for Region");
    }

    void* mem = nullptr;
    vkMapMemory(vkDev, devReg, 0, regSize, 0, &mem);
    memcpy(mem, &r, regSize);
    vkUnmapMemory(vkDev, devReg);

    /**
      * And again for the PixelFormat
      */

    VkBuffer fmtBuf;
    VkBufferCreateInfo fmtInfo{};
    fmtInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    fmtInfo.size = fmtSize;
    fmtInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    fmtInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if(vkCreateBuffer(vkDev, &fmtInfo, nullptr, &fmtBuf) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create device buffer for Region");
    }

    vkGetBufferMemoryRequirements(vkDev, fmtBuf, &memReqs);

    //Acquire and allocate the memory for the buffer
    memInfo.allocationSize = memReqs.size;

    foundMem = false;
    for( uint32_t k = 0; k < vkMProps.memoryTypeCount; k++)
    {
        const VkMemoryType memoryType = vkMProps.memoryTypes[k];

        if (  (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT & memoryType.propertyFlags)
           && (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT & memoryType.propertyFlags)
           && (memReqs.size < vkMProps.memoryHeaps[memoryType.heapIndex].size)      )
        {
            memInfo.memoryTypeIndex = k;
            foundMem = true;
            break;
        }
    }

    if(!foundMem)
    {
        throw std::runtime_error("Failed to find adequate memory for Region on the device");
    }

    VkDeviceMemory devFmt;
    if(vkAllocateMemory(vkDev, &memInfo, nullptr, &devFmt) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to allocate memory on device for Region");
    }

    if(vkBindBufferMemory(vkDev, fmtBuf, devFmt, 0) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to bind buffer memory for Region");
    }

    mem = nullptr;
    vkPixFormat fmt{*format};
    vkMapMemory(vkDev, devFmt, 0, fmtSize, 0, &mem);
    memcpy(mem, &fmt, fmtSize);
    vkUnmapMemory(vkDev,devFmt);


    /**
      * Create the descriptor set for the shader inputs
      * First - define the layout
      */

    //We're going to need a few of these, actually
    VkDescriptorSetLayoutBinding descSetLayoutBinding[3]{{},{},{}};
    //Output vector
    descSetLayoutBinding[0].binding = 0;
    descSetLayoutBinding[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descSetLayoutBinding[0].descriptorCount = 1;
    descSetLayoutBinding[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    //Region
    descSetLayoutBinding[1].binding = 1;
    descSetLayoutBinding[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descSetLayoutBinding[1].descriptorCount = 1;
    descSetLayoutBinding[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    //Pixel Format
    descSetLayoutBinding[2].binding = 2;
    descSetLayoutBinding[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descSetLayoutBinding[2].descriptorCount = 1;
    descSetLayoutBinding[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo descSetLayoutinfo{};
    descSetLayoutinfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descSetLayoutinfo.bindingCount = 3;
    descSetLayoutinfo.pBindings = descSetLayoutBinding;

    if(vkCreateDescriptorSetLayout(vkDev, &descSetLayoutinfo, nullptr, &vkDescLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create DescriptorSetLayout");
    }

    /**
     * Now allocate the actual descriptor set and pool
     */

    VkDescriptorPoolSize vkPoolSize{};
    vkPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    vkPoolSize.descriptorCount = 3;

    VkDescriptorPoolCreateInfo vkPoolInfo{};
    vkPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    vkPoolInfo.maxSets = 1;
    vkPoolInfo.poolSizeCount = 1;
    vkPoolInfo.pPoolSizes = &vkPoolSize;

    if(vkCreateDescriptorPool(vkDev, &vkPoolInfo, nullptr, &vkDescPool) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create DescriptorPool");
    }

    VkDescriptorSetAllocateInfo vkSetAll{};
    vkSetAll.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    vkSetAll.pSetLayouts = &vkDescLayout;
    vkSetAll.descriptorSetCount = 1;
    vkSetAll.descriptorPool = vkDescPool;

    if(vkAllocateDescriptorSets(vkDev, &vkSetAll, &vkDescSet) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to Allocate DescriptorSets");
    }



    VkDescriptorBufferInfo outDescBufInfo{};
    outDescBufInfo.buffer = outBuf;
    outDescBufInfo.offset = 0;
    outDescBufInfo.range = texSize;

    VkDescriptorBufferInfo regDescBufInfo{};
    regDescBufInfo.buffer = regBuf;
    regDescBufInfo.offset = 0;
    regDescBufInfo.range = regSize;

    VkDescriptorBufferInfo fmtDescBufInfo{};
    fmtDescBufInfo.buffer = fmtBuf;
    fmtDescBufInfo.offset = 0;
    fmtDescBufInfo.range = fmtSize;

    VkWriteDescriptorSet outWriteDescSet{};
    outWriteDescSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    outWriteDescSet.dstSet = vkDescSet;
    outWriteDescSet.dstBinding = 0; //binding to write to
    outWriteDescSet.descriptorCount = 1;
    outWriteDescSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    outWriteDescSet.pBufferInfo = &outDescBufInfo;

    VkWriteDescriptorSet regWriteDescSet{};
    regWriteDescSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    regWriteDescSet.dstSet = vkDescSet;
    regWriteDescSet.dstBinding = 1; //binding to write to
    regWriteDescSet.descriptorCount = 1;
    regWriteDescSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    regWriteDescSet.pBufferInfo = &regDescBufInfo;

    VkWriteDescriptorSet fmtWriteDescSet{};
    fmtWriteDescSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    fmtWriteDescSet.dstSet = vkDescSet;
    fmtWriteDescSet.dstBinding = 2; //binding to write to
    fmtWriteDescSet.descriptorCount = 1;
    fmtWriteDescSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    fmtWriteDescSet.pBufferInfo = &fmtDescBufInfo;



    vkUpdateDescriptorSets(vkDev, outWriteDescSet.descriptorCount, &outWriteDescSet, 0, nullptr);
    vkUpdateDescriptorSets(vkDev, regWriteDescSet.descriptorCount, &regWriteDescSet, 0, nullptr);
    vkUpdateDescriptorSets(vkDev, fmtWriteDescSet.descriptorCount, &fmtWriteDescSet, 0, nullptr);

    /**
     * Time to create the compute pipeline
     */

    uint32_t fileLength = 0;
    uint32_t* shaderCode = nullptr;

    shaderCode = readFile(fileLength, "FracGen.spv");


    VkShaderModuleCreateInfo vkShaderInfo{};
    vkShaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    vkShaderInfo.pCode = shaderCode;
    vkShaderInfo.codeSize = fileLength;

    if(vkCreateShaderModule(vkDev, &vkShaderInfo, nullptr, &vkShMod) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create ShaderModule");
    }
    if(shaderCode != nullptr)
    {
        delete[] shaderCode;
    }

    /**
     * Create the pipeline itself. For compute it is one stage: just the compute shader
     */

    VkPipelineShaderStageCreateInfo shStageCreate{};
    shStageCreate.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shStageCreate.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    shStageCreate.module = vkShMod;
    shStageCreate.pName = "main";

    VkPipelineLayoutCreateInfo ppLayoutCreate{};
    ppLayoutCreate.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    ppLayoutCreate.setLayoutCount = 1;
    ppLayoutCreate.pSetLayouts = &vkDescLayout;


    if(vkCreatePipelineLayout(vkDev, &ppLayoutCreate, nullptr, &vkPpLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create PipelineLayout");
    }

    VkComputePipelineCreateInfo pplCreate{};
    pplCreate.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pplCreate.stage = shStageCreate;
    pplCreate.layout = vkPpLayout;

    if(vkCreateComputePipelines(vkDev, VK_NULL_HANDLE, 1, &pplCreate, nullptr, &vkPpl) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create ComputePipeline");
    }

    /**
     * Almost there, time to create the command pool and the command buffer
     */

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = 0;
    poolInfo.queueFamilyIndex = vkQIdx; //This is important

    if(vkCreateCommandPool(vkDev, &poolInfo, nullptr, &vkCmdPool) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create CommandPool");
    }

    VkCommandBufferAllocateInfo cmdBufInfo{};
    cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdBufInfo.commandPool = vkCmdPool;
    // Primary buffers can be directly submitted
    // Secondary buffers are chained from other buffers
    cmdBufInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdBufInfo.commandBufferCount = 1;

    if(vkAllocateCommandBuffers(vkDev, &cmdBufInfo, &vkCmdBuffer) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to allocate CommandBuffer");
    }

    /**
     * Start actually using the command buffer (almost actually doing stuff)
     */

    VkCommandBufferBeginInfo cmdBeginInfo{};
    cmdBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cmdBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if(vkBeginCommandBuffer(vkCmdBuffer, &cmdBeginInfo) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to begin CommandBuffer");
    }


    //Bind the pipeline AND decriptor set to the command buffer

    vkCmdBindPipeline(vkCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, vkPpl);
    vkCmdBindDescriptorSets(vkCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, vkPpLayout, 0, 1, &vkDescSet, 0, nullptr);

    /*
     * Calling vkCmdDispatch basically starts the compute pipeline, and executes the compute shader.
     * The number of workgroups is specified in the arguments.
     */
    uint32_t x = static_cast<uint32_t>( ceil(width  / WORKGROUP_SIZE));
    uint32_t y = static_cast<uint32_t>( ceil(height / WORKGROUP_SIZE));
    uint32_t workgroupCount = x*y;
    vkCmdDispatch( vkCmdBuffer
                 , static_cast<uint32_t>( ceil(width  / WORKGROUP_SIZE))
                 , static_cast<uint32_t>( ceil(height / WORKGROUP_SIZE))
                 , 1);

    if(vkEndCommandBuffer(vkCmdBuffer) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to end CommandBuffer");
    }



    /**
     * At last, we actually get to run something
     */

    VkSubmitInfo subInfo{};
    subInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    subInfo.pCommandBuffers = &vkCmdBuffer;
    subInfo.commandBufferCount = 1;

    //We need a manual fence to make sure we wait for computation to end
    VkFence fence{};
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = 0;

    if(vkCreateFence(vkDev, &fenceInfo, nullptr, &fence) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create fence");
    }

    if(vkQueueSubmit(vkQ, 1, &subInfo, fence) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to submite command to queue");
    }

    if(vkWaitForFences(vkDev, 1, &fence, VK_TRUE, 100000000000) != VK_SUCCESS)
    {
        throw std::runtime_error("Error during shader run on the device");
    }
    vkDestroyFence(vkDev, fence, nullptr);

    /**
     * With the calculation done, time to retrieve the data
     */

    mem = nullptr;
    vkMapMemory(vkDev, devVec,0,texSize,0, &mem);
    memcpy(v,mem,texSize);
    vkUnmapMemory(vkDev,devVec);


    vkFreeCommandBuffers(vkDev, vkCmdPool, 1, &vkCmdBuffer);
    vkDestroyCommandPool(vkDev, vkCmdPool, nullptr);
    vkDestroyPipeline(vkDev,vkPpl, nullptr);
    vkDestroyPipelineLayout(vkDev, vkPpLayout, nullptr);
    vkDestroyShaderModule(vkDev, vkShMod, nullptr);
    vkDestroyDescriptorPool(vkDev,vkDescPool,nullptr);
    vkDestroyDescriptorSetLayout(vkDev,vkDescLayout,nullptr);
    vkFreeMemory(vkDev, devVec, nullptr);
    vkFreeMemory(vkDev, devReg, nullptr);
    vkFreeMemory(vkDev, devFmt, nullptr);
    vkDestroyBuffer(vkDev, outBuf, nullptr);
    vkDestroyBuffer(vkDev, regBuf, nullptr);
    vkDestroyBuffer(vkDev, fmtBuf, nullptr);

}

VKAPI_ATTR VkBool32 debugCallbackFn(    VkDebugReportFlagsEXT /*flags*/,
                                        VkDebugReportObjectTypeEXT /*objectType*/,
                                        uint64_t /*object*/,
                                        size_t /*location*/,
                                        int32_t /*messageCode*/,
                                        const char *pLayerPrefix,
                                        const char *pMessage,
                                        void */*pUserData*/                       )
{
    std::cout << "Vulkan Debug Report: "<< pLayerPrefix << ": " << pMessage << std::endl;
    return VK_FALSE;
}

FracGen::FracGen(bool bench)
{
    static bool once = false;
    //ALWAYS ALWAYS ALWAYS remember to empty initialise these structs!
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Toybrot Vulkan";
    appInfo.applicationVersion = VK_MAKE_VERSION(1,0,0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1,0,0);
    appInfo.apiVersion = VK_API_VERSION_1_1;


    // Check for validation layer
    std::vector<const char*> enabledLayers;
    std::vector<const char*> enabledExtensions;

    if(enableValidation)
    {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> layerProperties(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, layerProperties.data());

        /*
        And then we simply check if VK_LAYER_LUNARG_standard_validation is among the supported layers.
        */
        bool foundLayer = false;
        for (VkLayerProperties prop : layerProperties)
        {
            if (strcmp("VK_LAYER_LUNARG_standard_validation", prop.layerName) == 0)
            {
                foundLayer = true;
                break;
            }
        }

        if (!foundLayer)
        {
            throw std::runtime_error("Layer VK_LAYER_LUNARG_standard_validation not supported\n");
        }
        enabledLayers.push_back("VK_LAYER_LUNARG_standard_validation");

        uint32_t extensionCount = 0;

        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> extensions{extensionCount};

        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

        bool foundExtension = false;
        if(extensionCount > 0)
        {
            for(const auto& ext : extensions)
            {
                if (strcmp(VK_EXT_DEBUG_REPORT_EXTENSION_NAME, ext.extensionName) == 0)
                {
                    foundExtension = true;
                    break;
                }
            }
        }

        if (!foundExtension)
        {
            throw std::runtime_error("Extension VK_EXT_DEBUG_REPORT_EXTENSION_NAME not supported\n");
        }
        else
        {
            enabledExtensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
        }
    }

    //Time to create the instance

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo  = &appInfo;
    createInfo.enabledLayerCount = 0;

    createInfo.enabledLayerCount = static_cast<uint>(enabledLayers.size());
    createInfo.ppEnabledLayerNames = enabledLayers.data();
    createInfo.enabledExtensionCount = static_cast<uint>(enabledExtensions.size());
    createInfo.ppEnabledExtensionNames = enabledExtensions.data();

    if ( vkCreateInstance(&createInfo, nullptr, &vkInst) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create Vulkan instance");
    }

    if(enabledLayers.size() > 0)
    {
        VkDebugReportCallbackCreateInfoEXT cbcreateInfo = {};
        cbcreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
        cbcreateInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;
        cbcreateInfo.pfnCallback = &debugCallbackFn;

        // We have to explicitly load this function.
        auto vkCreateDebugReportCallbackEXT = reinterpret_cast<PFN_vkCreateDebugReportCallbackEXT>(vkGetInstanceProcAddr(vkInst, "vkCreateDebugReportCallbackEXT"));
        if (vkCreateDebugReportCallbackEXT == nullptr)
        {
            throw std::runtime_error("Could not load vkCreateDebugReportCallbackEXT");
        }

        // Create and register callback.
        if(vkCreateDebugReportCallbackEXT(vkInst, &cbcreateInfo, nullptr, &debugCallback) != VK_SUCCESS)
        {
            throw std::runtime_error("Could not create debug callback");
        }
    }



    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(vkInst, &deviceCount, nullptr);

    std::vector<VkPhysicalDevice> vkDevices{deviceCount};

    vkEnumeratePhysicalDevices( vkInst, &deviceCount, vkDevices.data());

    bool foundDevice = false;
    uint32_t deviceIndex = 0;

    if(deviceCount > 0)
    {
        VkPhysicalDeviceProperties props;
        for(uint32_t i = 0; (i < deviceCount) && (!foundDevice); i++)
        {
            vkGetPhysicalDeviceProperties(vkDevices[i], &props);

            uint32_t queueCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties( vkDevices[i]
                                                    , &queueCount
                                                    , nullptr     );

            std::vector< VkQueueFamilyProperties > queueProps{queueCount};

            vkGetPhysicalDeviceQueueFamilyProperties( vkDevices[i]
                                                    , &queueCount
                                                    , queueProps.data() );
            //Find a compute queue for the device;
            for(uint32_t j = 0; j < queueCount; j++)
            {
                if (  !(VK_QUEUE_GRAPHICS_BIT & queueProps[j].queueFlags)
                    && (VK_QUEUE_COMPUTE_BIT  & queueProps[j].queueFlags)  )
                {
                  deviceIndex = i;
                  vkQIdx = j;
                  foundDevice = true;
                  vkPhys = vkDevices[i];
                  break;
                }
            }

            if((foundDevice) && (!bench || !once) )
            {
                once = true;
                std::cout << deviceCount << " available Vulkan devices" << std::endl;

                std::cout << "Using device " << i << ":" << std::endl;
                std::cout << "Name: " << props.deviceName << std::endl;
                std::cout << "Supported version:" << VK_VERSION_MAJOR(props.apiVersion) << "."
                                                  << VK_VERSION_MINOR(props.apiVersion) << "."
                                                  << VK_VERSION_PATCH(props.apiVersion)
                                                  << std::endl;
                std::cout << "Max Workgroup size:" << props.limits.maxComputeWorkGroupSize[0]  << ","
                                                   << props.limits.maxComputeWorkGroupSize[1]  << ","
                                                   << props.limits.maxComputeWorkGroupSize[2]  << std::endl;
                std::cout << "Max Workgroup count:"<< props.limits.maxComputeWorkGroupCount[0] << ","
                                                   << props.limits.maxComputeWorkGroupCount[1] << ","
                                                   << props.limits.maxComputeWorkGroupCount[2] << std::endl;
}

        }
    }

    if(!foundDevice)
    {
        throw std::runtime_error("Could not get a Vulkan compute queue!");
    }

    float priority = 1.f;
    VkDeviceQueueCreateInfo queueInfo{};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = vkQIdx;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;

    VkPhysicalDeviceFeatures devFeatures{};
    devFeatures.shaderFloat64 = VK_TRUE;
    //devFeatures.logicOp = VK_TRUE;

    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    deviceInfo.pEnabledFeatures = &devFeatures;


    if(vkCreateDevice(vkPhys, &deviceInfo, nullptr, &vkDev) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create Vk Logical device");
    }

    vkGetDeviceQueue(vkDev, vkQIdx, 0, &vkQ);

    vkGetPhysicalDeviceMemoryProperties(vkPhys, &vkMProps);

}

FracGen::~FracGen()
{
    if (enableValidation)
    {
        // destroy callback.
        auto func = reinterpret_cast<PFN_vkDestroyDebugReportCallbackEXT>(vkGetInstanceProcAddr(vkInst, "vkDestroyDebugReportCallbackEXT"));
        if (func == nullptr)
        {
            throw std::runtime_error("Could not load vkDestroyDebugReportCallbackEXT");
        }
        func(vkInst, debugCallback, nullptr);
    }


    vkDestroyDevice(vkDev, nullptr);
    vkDestroyInstance(vkInst, nullptr);
}


bool operator==(const Region &r1, const Region &r2)
{
    return (   (r1.Imax - r2.Imax <= DBL_EPSILON) && (r1.Imin - r2.Imin <= DBL_EPSILON)
            && (r1.Rmax - r2.Rmax <= DBL_EPSILON) && (r1.Rmin - r2.Rmin <= DBL_EPSILON) );
}
