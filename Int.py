def combinationSum(nums: list, target: int) -> list:
        to_del = []
        for i in range(len(nums)):
            if nums[i] > target:
                to_del.append(i)
        for i in to_del[::-1]:
            nums.pop(i)
        return nums

print(combinationSum([7, 4, 5, 3], 3))