// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

contract Student {

 
    struct StudentData {
        string name;
        uint256 rollno;
    }

       StudentData[] public studentarr;

    // Function to add a student
    function addStudent(string memory name, uint256 rollno) public {
        for (uint i = 0; i < studentarr.length; i++) {
            if (studentarr[i].rollno == rollno) {
                revert("Student with this roll number already exists");
            }
        }
        studentarr.push(StudentData(name, rollno));
    }

    // Function to get the number of students
    function getStudentsLength() public view returns (uint) {
        return studentarr.length;
    }

    // Function to display all students
    function displayAllStudents() public view returns (StudentData[] memory) {
        return studentarr;
    }

    // Function to get a student by index
    function getStudentByIndex(uint idx) public view returns (StudentData memory) {
        require(idx < studentarr.length, "Index out of bound");
        return studentarr[idx];
    }

    // Fallback function

    fallback() external payable {
        // This function will handle any external function calls not present in our contract
    }

    // Receive function
    receive() external payable {
        // This function will handle Ether sent by an external user without any data
    }
}
