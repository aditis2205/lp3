// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

contract Bank {
    uint256 balance = 0;

    function deposit() public payable {
        require(msg.value > 0, "Deposit amount should be greater than 0.");
        balance += msg.value;
    }

    function withdraw(uint256 amount) public {
        require(balance >= amount, "Insufficient contract balance.");
        payable(msg.sender).transfer(amount);
        balance -= amount;
    }

    function showbalance() public view returns (uint256) {
        return balance;
    }
}
