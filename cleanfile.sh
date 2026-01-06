#!/bin/bash

# 1. 定义目标根目录（可根据你的实际路径修改，绝对路径/相对路径均可）
TARGET_DIR="./models"

# 2. 检查目标目录是否存在，不存在则报错退出
if [ ! -d "$TARGET_DIR" ]; then
    echo "错误：目标目录 $TARGET_DIR 不存在！"
    exit 1
fi

# 3. 遍历目标目录下的所有直接子文件夹（不递归遍历孙子文件夹）
# 注："$TARGET_DIR"/*/ 会自动匹配所有子文件夹（末尾/确保只匹配目录）
for sub_dir in "$TARGET_DIR"/*/; do
    # 去除目录路径末尾的斜杠（避免后续统计异常）
    sub_dir_clean=$(echo "$sub_dir" | sed 's/\/$//')

    # 4. 统计当前子文件夹内的【直接文件数量】（不包含子文件夹，-maxdepth 1 限制仅当前目录）
    # -type f 仅匹配文件（排除目录、链接等）
    file_count=$(find "$sub_dir_clean" -maxdepth 1 -type f | wc -l)

    # 5. 判断文件数量是否少于 3（0 个或 1, 2 个文件均满足条件）
    if [ "$file_count" -lt 3 ]; then
        echo "即将删除文件夹：$sub_dir_clean（内含文件数：$file_count）"
        # 6. 删除该文件夹及内部所有内容（-rf 强制递归删除，无交互提示）
        rm -rf "$sub_dir_clean"
        if [ $? -eq 0 ]; then
            echo "成功删除：$sub_dir_clean"
        else
            echo "失败：无法删除 $sub_dir_clean"
        fi
    fi
done

echo "====================================="
echo "文件夹清理任务执行完成！"
exit 0